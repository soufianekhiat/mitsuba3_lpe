/*
    OpenPBR Surface BSDF plugin for Mitsuba 3.

    Implements the Academy Software Foundation OpenPBR Surface material model
    with the following lobes:
      - Diffuse reflection (Lambert with Disney retro-reflection)
      - Specular reflection (GGX microfacet, blended metallic + dielectric)
      - Specular transmission (GGX microfacet refraction)
      - Coat (GGX microfacet)
      - Fuzz/sheen (Charlie NDF microfiber scattering)
      - Subsurface (diffuse color blend approximation)
      - Thin film iridescence (simplified Airy model)

    Parameters follow the OpenPBR Surface v1.1 specification.
    Reference: https://academysoftwarefoundation.github.io/OpenPBR/
*/

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/texture.h>

// Reuse Schlick helpers from principled BSDF
#include "principledhelpers.h"

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class OpenPBRSurface final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture, MicrofacetDistribution)

    OpenPBRSurface(const Properties &props) : Base(props) {
        // ---- Base layer ----
        m_base_weight    = props.get<ScalarFloat>("base_weight", 1.0f);
        m_base_color     = props.get_texture<Texture>("base_color", 0.8f);
        m_has_metalness  = get_flag("base_metalness", props);
        m_base_metalness = props.get_texture<Texture>("base_metalness", 0.0f);

        // ---- Specular layer ----
        m_specular_weight     = props.get<ScalarFloat>("specular_weight", 1.0f);
        m_specular_color      = props.get_texture<Texture>("specular_color", 1.0f);
        m_specular_roughness  = props.get_texture<Texture>("specular_roughness", 0.3f);
        m_specular_ior        = props.get<ScalarFloat>("specular_ior", 1.5f);
        m_has_anisotropy      = get_flag("specular_anisotropy", props);
        m_specular_anisotropy = props.get_texture<Texture>("specular_anisotropy", 0.0f);

        // ---- Transmission layer ----
        m_has_transmission    = get_flag("transmission_weight", props);
        m_transmission_weight = props.get_texture<Texture>("transmission_weight", 0.0f);
        m_transmission_color  = props.get_texture<Texture>("transmission_color", 1.0f);
        m_transmission_depth  = props.get<ScalarFloat>("transmission_depth", 0.0f);

        // ---- Subsurface layer ----
        m_has_subsurface      = get_flag("subsurface_weight", props);
        m_subsurface_weight   = props.get_texture<Texture>("subsurface_weight", 0.0f);
        m_subsurface_color    = props.get_texture<Texture>("subsurface_color", 0.8f);
        m_subsurface_radius   = props.get<ScalarFloat>("subsurface_radius", 1.0f);

        // ---- Fuzz/sheen layer ----
        m_has_fuzz       = get_flag("fuzz_weight", props);
        m_fuzz_weight    = props.get_texture<Texture>("fuzz_weight", 0.0f);
        m_fuzz_color     = props.get_texture<Texture>("fuzz_color", 1.0f);
        m_fuzz_roughness = props.get_texture<Texture>("fuzz_roughness", 0.5f);

        // ---- Coat layer ----
        m_has_coat       = get_flag("coat_weight", props);
        m_coat_weight    = props.get_texture<Texture>("coat_weight", 0.0f);
        m_coat_color     = props.get_texture<Texture>("coat_color", 1.0f);
        m_coat_roughness = props.get_texture<Texture>("coat_roughness", 0.0f);
        m_coat_ior       = props.get<ScalarFloat>("coat_ior", 1.5f);

        // ---- Thin film ----
        m_thin_film_thickness = props.get<ScalarFloat>("thin_film_thickness", 0.0f);
        m_thin_film_ior       = props.get<ScalarFloat>("thin_film_ior", 1.5f);
        m_has_thin_film       = m_thin_film_thickness > 0.0f;

        // ---- Geometry ----
        m_thin_walled = props.get<bool>("geometry_thin_walled", false);

        // Ensure eta > 1 for non-thin-walled transmission
        if (m_has_transmission && m_specular_ior == 1.0f)
            m_specular_ior = 1.001f;

        initialize_lobes();
        dr::make_opaque(m_specular_ior, m_coat_ior, m_base_weight,
                        m_specular_weight, m_thin_film_thickness,
                        m_thin_film_ior, m_transmission_depth,
                        m_subsurface_radius);
    }

    void initialize_lobes() {
        m_components.clear();

        // Component 0: Diffuse reflection
        m_components.push_back(BSDFFlags::DiffuseReflection |
                               BSDFFlags::FrontSide);

        // Component 1: Main specular reflection (GGX)
        uint32_t spec = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide |
                        BSDFFlags::BackSide;
        if (m_has_anisotropy)
            spec = spec | BSDFFlags::Anisotropic;
        m_components.push_back(spec);

        // Component 2: Specular transmission (optional)
        if (m_has_transmission) {
            uint32_t trans = BSDFFlags::GlossyTransmission |
                             BSDFFlags::FrontSide | BSDFFlags::BackSide |
                             BSDFFlags::NonSymmetric;
            if (m_has_anisotropy)
                trans = trans | BSDFFlags::Anisotropic;
            m_components.push_back(trans);
        }

        // Coat reflection (optional)
        if (m_has_coat) {
            m_components.push_back(BSDFFlags::GlossyReflection |
                                   BSDFFlags::FrontSide);
        }

        // Fuzz reflection (optional)
        if (m_has_fuzz) {
            m_components.push_back(BSDFFlags::DiffuseReflection |
                                   BSDFFlags::FrontSide);
        }

        for (auto c : m_components)
            m_flags |= c;
    }

    // ---- Charlie NDF for fuzz/sheen lobe ----
    // D_charlie(alpha, sin_theta_h) = (2 + 1/alpha) / (2*pi) * sin^(1/alpha)(theta_h)
    Float charlie_D(Float sin_theta_h, Float alpha) const {
        Float inv_alpha = dr::rcp(alpha);
        return (2.0f + inv_alpha) * dr::InvTwoPi<Float> *
               dr::pow(dr::maximum(sin_theta_h, 1e-6f), inv_alpha);
    }

    // Neubelt visibility for cloth/sheen
    Float neubelt_V(Float cos_i, Float cos_o) const {
        return dr::rcp(4.0f * (dr::abs(cos_i) + dr::abs(cos_o) -
               dr::abs(cos_i) * dr::abs(cos_o)));
    }

    // ---- Thin film iridescence (simplified Airy) ----
    // Returns a spectral modifier for Fresnel based on thin film interference
    UnpolarizedSpectrum thin_film_factor(Float cos_theta) const {
        if (!m_has_thin_film)
            return UnpolarizedSpectrum(1.0f);

        Float sin_theta = dr::safe_sqrt(1.0f - dr::square(cos_theta));
        Float sin_film = sin_theta / Float(m_thin_film_ior);
        Float cos_film = dr::safe_sqrt(1.0f - dr::square(sin_film));
        Float opd = 2.0f * Float(m_thin_film_ior) *
                    Float(m_thin_film_thickness) * cos_film;

        // Phase for RGB wavelengths (nm)
        Float ph_r = 2.0f * dr::Pi<Float> * opd / 650.0f;
        Float ph_g = 2.0f * dr::Pi<Float> * opd / 550.0f;
        Float ph_b = 2.0f * dr::Pi<Float> * opd / 450.0f;

        // Interference factor: modulates reflectance per channel
        UnpolarizedSpectrum factor;
        factor[0] = dr::maximum(0.0f, 0.5f + 0.5f * dr::cos(ph_r));
        factor[1] = dr::maximum(0.0f, 0.5f + 0.5f * dr::cos(ph_g));
        factor[2] = dr::maximum(0.0f, 0.5f + 0.5f * dr::cos(ph_b));
        return factor;
    }

    // ---- Transmission color with depth ----
    // For solid objects: Beer's law approximation at single interface
    // For thin_walled: direct color (no volume)
    UnpolarizedSpectrum effective_trans_color(
            const UnpolarizedSpectrum &trans_color) const {
        if (m_thin_walled)
            return trans_color;
        // Beer's law: at each interface apply pow(color, 0.5/depth)
        // Default depth=0 means use sqrt(color) as before
        if (m_transmission_depth > 0.0f) {
            Float exponent = 0.5f / dr::maximum(Float(m_transmission_depth), 0.001f);
            return dr::pow(dr::maximum(trans_color, 0.0f), exponent);
        }
        return dr::sqrt(dr::maximum(trans_color, 0.0f));
    }

    // ---- Effective IOR for refraction direction ----
    // thin_walled: nearly 1 (straight through, no bending)
    // solid: actual specular_ior
    ScalarFloat refraction_ior() const {
        return m_thin_walled ? 1.001f : m_specular_ior;
    }

    std::pair<BSDFSample3f, Spectrum>
    sample(const BSDFContext &ctx, const SurfaceInteraction3f &si,
           Float sample1, const Point2f &sample2, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs   = dr::zeros<BSDFSample3f>();

        active &= cos_theta_i != 0.0f;
        if (unlikely(dr::none_or<false>(active)))
            return { bs, 0.0f };

        // Evaluate texture parameters
        Float metalness =
            m_has_metalness ? m_base_metalness->eval_1(si, active) : 0.0f;
        Float roughness   = m_specular_roughness->eval_1(si, active);
        Float anisotropy  =
            m_has_anisotropy ? m_specular_anisotropy->eval_1(si, active) : 0.0f;
        Float trans_weight =
            m_has_transmission ? m_transmission_weight->eval_1(si, active) : 0.0f;
        Float coat_weight =
            m_has_coat ? m_coat_weight->eval_1(si, active) : 0.0f;
        Float fuzz_weight =
            m_has_fuzz ? m_fuzz_weight->eval_1(si, active) : 0.0f;

        Mask front_side = cos_theta_i > 0.0f;

        // BRDF weight (fraction going to diffuse) and BTDF weight
        Float brdf = m_base_weight * (1.0f - metalness) * (1.0f - trans_weight);
        Float btdf = m_has_transmission ? (1.0f - metalness) * trans_weight : 0.0f;

        // Compute specular roughness -> alpha
        auto [ax, ay] = calc_dist_params(anisotropy, roughness, m_has_anisotropy);
        MicrofacetDistribution spec_distr(MicrofacetType::GGX, ax, ay);

        // Sample microfacet normal for specular/transmission
        Normal3f m_spec = std::get<0>(
            spec_distr.sample(dr::mulsign(si.wi, cos_theta_i), sample2));

        // Fresnel at the sampled microfacet (use real IOR for Fresnel)
        auto [F_dielectric, cos_theta_t, eta_it, eta_ti] =
            fresnel(dr::dot(si.wi, m_spec), Float(m_specular_ior));

        // For refraction direction, use effective IOR (1.001 for thin_walled)
        ScalarFloat ior_refract = refraction_ior();
        auto [F_refract, cos_t_refr, eta_it_r, eta_ti_r] =
            fresnel(dr::dot(si.wi, m_spec), Float(ior_refract));

        // Inside the material: only specular reflection and transmission
        active &= (front_side || (btdf > 0.0f));

        // ---- Sampling probabilities ----
        Float prob_spec = dr::select(
            front_side,
            1.0f - btdf * (1.0f - F_dielectric),
            F_dielectric);
        Float prob_trans = m_has_transmission
            ? dr::select(front_side,
                         btdf * (1.0f - F_dielectric),
                         (1.0f - F_dielectric))
            : 0.0f;
        Float prob_coat  = m_has_coat
            ? dr::select(front_side, 0.25f * coat_weight, 0.0f) : 0.0f;
        Float prob_fuzz  = m_has_fuzz
            ? dr::select(front_side, 0.5f * fuzz_weight, 0.0f) : 0.0f;
        Float prob_diff  = dr::select(front_side, brdf, 0.0f);

        // Normalize
        Float rcp_total = dr::rcp(prob_spec + prob_trans + prob_coat +
                                  prob_fuzz + prob_diff);
        prob_spec  *= rcp_total;
        prob_trans *= rcp_total;
        prob_coat  *= rcp_total;
        prob_fuzz  *= rcp_total;
        prob_diff  *= rcp_total;

        // Lobe selection via sample1
        Float curr(0.0f);
        Mask sample_diff = active && (sample1 < prob_diff);
        curr += prob_diff;
        Mask sample_fuzz = m_has_fuzz && active &&
            (sample1 >= curr) && (sample1 < curr + prob_fuzz);
        curr += prob_fuzz;
        Mask sample_coat = m_has_coat && active &&
            (sample1 >= curr) && (sample1 < curr + prob_coat);
        curr += prob_coat;
        Mask sample_trans = m_has_transmission && active &&
            (sample1 >= curr) && (sample1 < curr + prob_trans);
        curr += prob_trans;
        Mask sample_spec = active && (sample1 >= curr);

        bs.eta = 1.0f;

        // ---- Main specular reflection ----
        if (dr::any_or<true>(sample_spec)) {
            Vector3f wo = reflect(si.wi, m_spec);
            dr::masked(bs.wo, sample_spec) = wo;
            dr::masked(bs.sampled_component, sample_spec) = 1;
            dr::masked(bs.sampled_type, sample_spec) =
                +BSDFFlags::GlossyReflection;

            Mask valid_reflect = cos_theta_i * Frame3f::cos_theta(wo) > 0.0f;
            active &= (!sample_spec ||
                (mac_mic_compatibility(Vector3f(m_spec), si.wi, wo,
                                       cos_theta_i, true) && valid_reflect));
        }

        // ---- Specular transmission ----
        if (m_has_transmission && dr::any_or<true>(sample_trans)) {
            // Use effective IOR for refraction direction
            Vector3f wo = refract(si.wi, m_spec, cos_t_refr, eta_ti_r);
            dr::masked(bs.wo, sample_trans) = wo;
            dr::masked(bs.sampled_component, sample_trans) = 2;
            dr::masked(bs.sampled_type, sample_trans) =
                +BSDFFlags::GlossyTransmission;
            dr::masked(bs.eta, sample_trans) = eta_it_r;

            Mask valid_refract = cos_theta_i * Frame3f::cos_theta(wo) < 0.0f;
            active &= (!sample_trans ||
                (mac_mic_compatibility(Vector3f(m_spec), si.wi, wo,
                                       cos_theta_i, false) && valid_refract));
        }

        // ---- Coat reflection ----
        if (m_has_coat && dr::any_or<true>(sample_coat)) {
            Float coat_rough = m_coat_roughness->eval_1(si, active);
            Float coat_alpha = dr::maximum(0.001f, dr::square(coat_rough));
            MicrofacetDistribution coat_distr(MicrofacetType::GGX,
                                              coat_alpha, coat_alpha);
            Normal3f m_coat = std::get<0>(coat_distr.sample(si.wi, sample2));
            Vector3f wo = reflect(si.wi, m_coat);

            dr::masked(bs.wo, sample_coat) = wo;
            dr::masked(bs.sampled_component, sample_coat) = 3;
            dr::masked(bs.sampled_type, sample_coat) =
                +BSDFFlags::GlossyReflection;

            Mask valid_reflect = cos_theta_i * Frame3f::cos_theta(wo) > 0.0f;
            active &= (!sample_coat ||
                (mac_mic_compatibility(Vector3f(m_coat), si.wi, wo,
                                       cos_theta_i, true) && valid_reflect));
        }

        // ---- Fuzz (cosine hemisphere sampling) ----
        if (m_has_fuzz && dr::any_or<true>(sample_fuzz)) {
            Vector3f wo = warp::square_to_cosine_hemisphere(sample2);
            dr::masked(bs.wo, sample_fuzz) = wo;
            dr::masked(bs.sampled_component, sample_fuzz) = 4;
            dr::masked(bs.sampled_type, sample_fuzz) =
                +BSDFFlags::DiffuseReflection;

            Mask valid_reflect = cos_theta_i * Frame3f::cos_theta(wo) > 0.0f;
            active &= (!sample_fuzz || valid_reflect);
        }

        // ---- Cosine hemisphere (diffuse) ----
        if (dr::any_or<true>(sample_diff)) {
            Vector3f wo = warp::square_to_cosine_hemisphere(sample2);
            dr::masked(bs.wo, sample_diff) = wo;
            dr::masked(bs.sampled_component, sample_diff) = 0;
            dr::masked(bs.sampled_type, sample_diff) =
                +BSDFFlags::DiffuseReflection;

            Mask valid_reflect = cos_theta_i * Frame3f::cos_theta(wo) > 0.0f;
            active &= (!sample_diff || valid_reflect);
        }

        bs.pdf = pdf(ctx, si, bs.wo, active);
        active &= bs.pdf > 0.0f;
        Spectrum result = eval(ctx, si, bs.wo, active);
        return { bs, result / bs.pdf & active };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i != 0.0f;
        if (unlikely(dr::none_or<false>(active)))
            return 0.0f;

        // Evaluate texture parameters
        Float metalness =
            m_has_metalness ? m_base_metalness->eval_1(si, active) : 0.0f;
        Float roughness  = m_specular_roughness->eval_1(si, active);
        Float anisotropy =
            m_has_anisotropy ? m_specular_anisotropy->eval_1(si, active) : 0.0f;
        Float trans_weight =
            m_has_transmission ? m_transmission_weight->eval_1(si, active) : 0.0f;
        Float coat_weight =
            m_has_coat ? m_coat_weight->eval_1(si, active) : 0.0f;
        Float fuzz_weight =
            m_has_fuzz ? m_fuzz_weight->eval_1(si, active) : 0.0f;
        UnpolarizedSpectrum base_color = m_base_color->eval(si, active);

        // Subsurface: blend diffuse color
        UnpolarizedSpectrum diffuse_color = base_color;
        if (m_has_subsurface) {
            Float ss_weight = m_subsurface_weight->eval_1(si, active);
            UnpolarizedSpectrum ss_color = m_subsurface_color->eval(si, active);
            diffuse_color = (1.0f - ss_weight) * base_color +
                            ss_weight * ss_color;
        }

        // Lobe weights
        Float brdf = m_base_weight * (1.0f - metalness) * (1.0f - trans_weight);
        Float btdf = (1.0f - metalness) * trans_weight;

        Float cos_theta_o = Frame3f::cos_theta(wo);
        Mask reflect  = cos_theta_i * cos_theta_o > 0.0f;
        Mask refract  = cos_theta_i * cos_theta_o < 0.0f;
        Mask front_side = cos_theta_i > 0.0f;

        // Effective IOR for refraction direction
        ScalarFloat ior_refract = refraction_ior();
        Float inv_eta_r  = dr::rcp(Float(ior_refract));
        Float eta_path_r = dr::select(front_side, Float(ior_refract), inv_eta_r);
        Float inv_eta_path_r = dr::select(front_side, inv_eta_r, Float(ior_refract));

        // Main specular distribution
        auto [ax, ay] = calc_dist_params(anisotropy, roughness, m_has_anisotropy);
        MicrofacetDistribution spec_distr(MicrofacetType::GGX, ax, ay);

        // Halfway vector (uses refraction IOR for transmission half-vector)
        Vector3f wh = dr::normalize(
            si.wi + wo * dr::select(reflect, 1.0f, eta_path_r));
        wh = dr::mulsign(wh, Frame3f::cos_theta(wh));

        // Dielectric Fresnel (use real IOR for Fresnel amplitude)
        auto [F_dielectric, cos_theta_t, eta_it, eta_ti] =
            fresnel(dr::dot(si.wi, wh), Float(m_specular_ior));

        // Thin film: modulate dielectric Fresnel
        UnpolarizedSpectrum film_mod = thin_film_factor(dr::abs(dr::dot(si.wi, wh)));

        // Compatibility masks
        Mask reflect_compat =
            mac_mic_compatibility(wh, si.wi, wo, cos_theta_i, true);
        Mask refract_compat =
            mac_mic_compatibility(wh, si.wi, wo, cos_theta_i, false);

        // Active masks for each lobe
        Mask spec_active = active && reflect && reflect_compat &&
                           (F_dielectric > 0.0f);
        Mask trans_active = m_has_transmission && active && (btdf > 0.0f) &&
                            refract && refract_compat &&
                            (F_dielectric < 1.0f);
        Mask diff_active  = active && (brdf > 0.0f) && reflect && front_side;
        Mask coat_active  = m_has_coat && active && (coat_weight > 0.0f) &&
                            reflect && reflect_compat && front_side;
        Mask fuzz_active  = m_has_fuzz && active && (fuzz_weight > 0.0f) &&
                            reflect && front_side;

        // Evaluate NDF and shadowing-masking
        Float D = spec_distr.eval(wh);
        Float G = spec_distr.G(si.wi, wo, wh);

        UnpolarizedSpectrum value(0.0f);

        // ---- Main specular reflection ----
        if (dr::any_or<true>(spec_active)) {
            Float cos_d = dr::dot(si.wi, wh);
            UnpolarizedSpectrum spec_color = m_specular_color->eval(si, active);

            // Metallic Fresnel (Schlick with base_color as F0)
            UnpolarizedSpectrum F_metallic =
                calc_schlick<UnpolarizedSpectrum>(base_color, cos_d,
                                                  Float(m_specular_ior));

            // Dielectric Fresnel (scalar, tinted by specular_color + thin film)
            UnpolarizedSpectrum F_dielectric_tinted =
                F_dielectric * m_specular_weight * spec_color * film_mod;

            // Blend metallic and dielectric
            UnpolarizedSpectrum F_combined =
                metalness * F_metallic +
                (1.0f - metalness) * F_dielectric_tinted;

            // f * cos_o = F * D * G / (4 * |cos_i|)
            dr::masked(value, spec_active) +=
                F_combined * D * G / (4.0f * dr::abs(cos_theta_i));
        }

        // ---- Specular transmission ----
        if (m_has_transmission && dr::any_or<true>(trans_active)) {
            Float scale = (ctx.mode == TransportMode::Radiance)
                ? dr::square(inv_eta_path_r) : Float(1.0f);

            UnpolarizedSpectrum trans_color =
                m_transmission_color->eval(si, active);
            UnpolarizedSpectrum eff_color = effective_trans_color(trans_color);

            // Refraction half-vector uses refraction IOR
            auto [F_refr, ct_r, eit_r, eti_r] =
                fresnel(dr::dot(si.wi, wh), Float(ior_refract));

            dr::masked(value, trans_active) +=
                eff_color * btdf *
                dr::abs((scale * (1.0f - F_refr) * D * G * eta_path_r *
                eta_path_r * dr::dot(si.wi, wh) * dr::dot(wo, wh)) /
                (cos_theta_i * dr::square(dr::dot(si.wi, wh) +
                eta_path_r * dr::dot(wo, wh))));
        }

        // ---- Coat reflection ----
        if (m_has_coat && dr::any_or<true>(coat_active)) {
            Float coat_rough = m_coat_roughness->eval_1(si, active);
            Float coat_alpha = dr::maximum(0.001f, dr::square(coat_rough));
            MicrofacetDistribution coat_distr(MicrofacetType::GGX,
                                              coat_alpha, coat_alpha);

            Float D_coat = coat_distr.eval(wh);
            Float G_coat = coat_distr.G(si.wi, wo, wh);
            Float F_coat = std::get<0>(
                fresnel(dr::dot(si.wi, wh), Float(m_coat_ior)));

            UnpolarizedSpectrum coat_col = m_coat_color->eval(si, active);

            // f * cos_o = coat * F * D * G / (4 * |cos_i|)
            dr::masked(value, coat_active) +=
                coat_weight * coat_col * F_coat * D_coat * G_coat /
                (4.0f * dr::abs(cos_theta_i));
        }

        // ---- Fuzz/sheen reflection ----
        if (m_has_fuzz && dr::any_or<true>(fuzz_active)) {
            Float fuzz_rough = m_fuzz_roughness->eval_1(si, active);
            Float fuzz_alpha = dr::maximum(0.01f, dr::square(fuzz_rough));
            UnpolarizedSpectrum fuzz_col = m_fuzz_color->eval(si, active);

            // Charlie NDF
            Float sin_theta_h = dr::safe_sqrt(
                1.0f - dr::square(Frame3f::cos_theta(wh)));
            Float D_fuzz = charlie_D(sin_theta_h, fuzz_alpha);

            // Neubelt visibility
            Float V_fuzz = neubelt_V(cos_theta_i, cos_theta_o);

            // f * cos_o = fuzz_weight * fuzz_color * D * V * |cos_o|
            dr::masked(value, fuzz_active) +=
                fuzz_weight * fuzz_col * D_fuzz * V_fuzz *
                dr::abs(cos_theta_o);
        }

        // ---- Diffuse reflection ----
        if (dr::any_or<true>(diff_active)) {
            Float Fo = schlick_weight(dr::abs(cos_theta_o));
            Float Fi = schlick_weight(dr::abs(cos_theta_i));

            // Disney diffuse with retro-reflection
            Float f_diff = (1.0f - 0.5f * Fi) * (1.0f - 0.5f * Fo);
            Float cos_theta_d = dr::dot(wh, wo);
            Float Rr = 2.0f * roughness * dr::square(cos_theta_d);
            Float f_retro = Rr * (Fo + Fi + Fo * Fi * (Rr - 1.0f));

            // f * cos_o = brdf * diffuse_color * (f_diff + f_retro) * |cos_o| / pi
            dr::masked(value, diff_active) +=
                brdf * dr::abs(cos_theta_o) * diffuse_color *
                dr::InvPi<Float> * (f_diff + f_retro);
        }

        return depolarizer<Spectrum>(value) & active;
    }

    Float pdf(const BSDFContext &, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i != 0.0f;
        if (unlikely(dr::none_or<false>(active)))
            return 0.0f;

        // Evaluate texture parameters
        Float metalness =
            m_has_metalness ? m_base_metalness->eval_1(si, active) : 0.0f;
        Float roughness  = m_specular_roughness->eval_1(si, active);
        Float anisotropy =
            m_has_anisotropy ? m_specular_anisotropy->eval_1(si, active) : 0.0f;
        Float trans_weight =
            m_has_transmission ? m_transmission_weight->eval_1(si, active) : 0.0f;
        Float coat_weight =
            m_has_coat ? m_coat_weight->eval_1(si, active) : 0.0f;
        Float fuzz_weight =
            m_has_fuzz ? m_fuzz_weight->eval_1(si, active) : 0.0f;

        Float brdf = m_base_weight * (1.0f - metalness) * (1.0f - trans_weight);
        Float btdf = (1.0f - metalness) * trans_weight;

        Mask front_side = cos_theta_i > 0.0f;
        ScalarFloat ior_refract = refraction_ior();
        Float eta_path = dr::select(front_side, Float(ior_refract),
                                    dr::rcp(Float(ior_refract)));
        Float cos_theta_o = Frame3f::cos_theta(wo);
        Mask reflect = cos_theta_i * cos_theta_o > 0.0f;
        Mask refract = cos_theta_i * cos_theta_o < 0.0f;

        // Specular distribution
        auto [ax, ay] = calc_dist_params(anisotropy, roughness, m_has_anisotropy);
        MicrofacetDistribution spec_distr(MicrofacetType::GGX, ax, ay);

        // Halfway vector
        Vector3f wh = dr::normalize(
            si.wi + wo * dr::select(reflect, Float(1.0f), eta_path));
        wh = dr::mulsign(wh, Frame3f::cos_theta(wh));

        // Dielectric Fresnel (use real IOR for probability weighting)
        auto [F_dielectric, cos_theta_t, eta_it, eta_ti] =
            fresnel(dr::dot(si.wi, wh), Float(m_specular_ior));

        // Probabilities (must match sample())
        Float prob_spec = dr::select(
            front_side,
            1.0f - btdf * (1.0f - F_dielectric),
            F_dielectric);
        Float prob_trans = m_has_transmission
            ? dr::select(front_side,
                         btdf * (1.0f - F_dielectric),
                         (1.0f - F_dielectric))
            : 0.0f;
        Float prob_coat = m_has_coat
            ? dr::select(front_side, 0.25f * coat_weight, 0.0f) : 0.0f;
        Float prob_fuzz = m_has_fuzz
            ? dr::select(front_side, 0.5f * fuzz_weight, 0.0f) : 0.0f;
        Float prob_diff = dr::select(front_side, brdf, 0.0f);

        Float rcp_total = dr::rcp(prob_spec + prob_trans + prob_coat +
                                  prob_fuzz + prob_diff);
        prob_spec  *= rcp_total;
        prob_trans *= rcp_total;
        prob_coat  *= rcp_total;
        prob_fuzz  *= rcp_total;
        prob_diff  *= rcp_total;

        // Jacobian |dwh/dwo|
        Float dwh_dwo_abs;
        if (m_has_transmission) {
            Float dot_wi_h = dr::dot(si.wi, wh);
            Float dot_wo_h = dr::dot(wo, wh);
            dwh_dwo_abs = dr::abs(
                dr::select(reflect, dr::rcp(4.0f * dot_wo_h),
                           (dr::square(eta_path) * dot_wo_h) /
                           dr::square(dot_wi_h + eta_path * dot_wo_h)));
        } else {
            dwh_dwo_abs = dr::abs(dr::rcp(4.0f * dr::dot(wo, wh)));
        }

        Float pdf(0.0f);

        // Macro-micro compatibility masks
        Mask mfacet_reflect_compat =
            mac_mic_compatibility(wh, si.wi, wo, cos_theta_i, true) && reflect;

        // Main specular reflection PDF
        dr::masked(pdf, mfacet_reflect_compat) +=
            prob_spec *
            spec_distr.pdf(dr::mulsign(si.wi, cos_theta_i), wh) * dwh_dwo_abs;

        // Diffuse PDF
        dr::masked(pdf, reflect) +=
            prob_diff * warp::square_to_cosine_hemisphere_pdf(wo);

        // Fuzz PDF (cosine hemisphere)
        if (m_has_fuzz) {
            dr::masked(pdf, reflect) +=
                prob_fuzz * warp::square_to_cosine_hemisphere_pdf(wo);
        }

        // Transmission PDF
        if (m_has_transmission) {
            Mask mfacet_trans_compat =
                mac_mic_compatibility(wh, si.wi, wo, cos_theta_i, false) &&
                refract;
            dr::masked(pdf, mfacet_trans_compat) +=
                prob_trans *
                spec_distr.pdf(dr::mulsign(si.wi, cos_theta_i), wh) *
                dwh_dwo_abs;
        }

        // Coat PDF
        if (m_has_coat) {
            Float coat_rough = m_coat_roughness->eval_1(si, active);
            Float coat_alpha = dr::maximum(0.001f, dr::square(coat_rough));
            MicrofacetDistribution coat_distr(MicrofacetType::GGX,
                                              coat_alpha, coat_alpha);
            dr::masked(pdf, mfacet_reflect_compat) +=
                prob_coat *
                coat_distr.pdf(dr::mulsign(si.wi, cos_theta_i), wh) *
                dwh_dwo_abs;
        }

        return pdf;
    }

    Spectrum eval_diffuse_reflectance(const SurfaceInteraction3f &si,
                                      Mask active) const override {
        Float metalness =
            m_has_metalness ? m_base_metalness->eval_1(si, active) : 0.0f;
        UnpolarizedSpectrum base_color = m_base_color->eval(si, active);

        // Subsurface blending for diffuse reflectance
        UnpolarizedSpectrum diffuse_color = base_color;
        if (m_has_subsurface) {
            Float ss_weight = m_subsurface_weight->eval_1(si, active);
            UnpolarizedSpectrum ss_color = m_subsurface_color->eval(si, active);
            diffuse_color = (1.0f - ss_weight) * base_color +
                            ss_weight * ss_color;
        }

        return diffuse_color * m_base_weight * (1.0f - metalness);
    }

    Mask has_attribute(const std::string &name,
                       Mask active = true) const override {
        if (name == "specular_reflectance" || name == "alpha" ||
            name == "roughness" || name == "eta")
            return active;
        return Base::has_attribute(name, active);
    }

    Float eval_attribute_1(const std::string &name,
                            const SurfaceInteraction3f &si,
                            Mask active = true) const override {
        if (name == "alpha") {
            Float r = m_specular_roughness->eval_1(si, active);
            return dr::square(r);
        }
        if (name == "roughness")
            return m_specular_roughness->eval_1(si, active);
        if (name == "eta")
            return Float(m_specular_ior);
        return Base::eval_attribute_1(name, si, active);
    }

    Color3f eval_attribute_3(const std::string &name,
                              const SurfaceInteraction3f &si,
                              Mask active = true) const override {
        if (name == "specular_reflectance") {
            Float metalness =
                m_has_metalness ? m_base_metalness->eval_1(si, active) : 0.0f;
            UnpolarizedSpectrum base_color = m_base_color->eval(si, active);
            Float F0 = schlick_R0_eta(Float(m_specular_ior));
            UnpolarizedSpectrum spec_color = m_specular_color->eval(si, active);
            UnpolarizedSpectrum result =
                metalness * base_color +
                (1.0f - metalness) * F0 * m_specular_weight * spec_color;
            return Color3f(result[0], result[1], result[2]);
        }
        return Base::eval_attribute_3(name, si, active);
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("base_color",           m_base_color,           ParamFlags::Differentiable);
        cb->put("base_metalness",       m_base_metalness,       ParamFlags::Differentiable);
        cb->put("specular_color",       m_specular_color,       ParamFlags::Differentiable);
        cb->put("specular_roughness",   m_specular_roughness,   ParamFlags::Differentiable | ParamFlags::Discontinuous);
        cb->put("specular_anisotropy",  m_specular_anisotropy,  ParamFlags::Differentiable);
        cb->put("specular_ior",         m_specular_ior,         ParamFlags::Differentiable | ParamFlags::Discontinuous);
        cb->put("transmission_weight",  m_transmission_weight,  ParamFlags::Differentiable);
        cb->put("transmission_color",   m_transmission_color,   ParamFlags::Differentiable);
        cb->put("subsurface_weight",    m_subsurface_weight,    ParamFlags::Differentiable);
        cb->put("subsurface_color",     m_subsurface_color,     ParamFlags::Differentiable);
        cb->put("fuzz_weight",          m_fuzz_weight,          ParamFlags::Differentiable);
        cb->put("fuzz_color",           m_fuzz_color,           ParamFlags::Differentiable);
        cb->put("fuzz_roughness",       m_fuzz_roughness,       ParamFlags::Differentiable);
        cb->put("coat_weight",          m_coat_weight,          ParamFlags::Differentiable);
        cb->put("coat_color",           m_coat_color,           ParamFlags::Differentiable);
        cb->put("coat_roughness",       m_coat_roughness,       ParamFlags::Differentiable);
        cb->put("coat_ior",             m_coat_ior,             ParamFlags::Differentiable | ParamFlags::Discontinuous);
    }

    void parameters_changed(const std::vector<std::string> &keys = {}) override {
        if (string::contains(keys, "base_metalness"))
            m_has_metalness = true;
        if (string::contains(keys, "transmission_weight"))
            m_has_transmission = true;
        if (string::contains(keys, "coat_weight"))
            m_has_coat = true;
        if (string::contains(keys, "specular_anisotropy"))
            m_has_anisotropy = true;
        if (string::contains(keys, "subsurface_weight"))
            m_has_subsurface = true;
        if (string::contains(keys, "fuzz_weight"))
            m_has_fuzz = true;

        if (m_specular_ior == 1.0f && m_has_transmission)
            m_specular_ior = 1.001f;

        initialize_lobes();
        dr::make_opaque(m_specular_ior, m_coat_ior, m_base_weight,
                        m_specular_weight, m_thin_film_thickness,
                        m_thin_film_ior, m_transmission_depth,
                        m_subsurface_radius);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "OpenPBRSurface[" << std::endl
            << "  base_weight = " << m_base_weight << "," << std::endl
            << "  base_color = " << m_base_color << "," << std::endl
            << "  base_metalness = " << m_base_metalness << "," << std::endl
            << "  specular_weight = " << m_specular_weight << "," << std::endl
            << "  specular_roughness = " << m_specular_roughness << "," << std::endl
            << "  specular_ior = " << m_specular_ior << "," << std::endl
            << "  transmission_weight = " << m_transmission_weight << "," << std::endl
            << "  transmission_color = " << m_transmission_color << "," << std::endl
            << "  transmission_depth = " << m_transmission_depth << "," << std::endl
            << "  subsurface_weight = " << m_subsurface_weight << "," << std::endl
            << "  subsurface_color = " << m_subsurface_color << "," << std::endl
            << "  fuzz_weight = " << m_fuzz_weight << "," << std::endl
            << "  fuzz_color = " << m_fuzz_color << "," << std::endl
            << "  fuzz_roughness = " << m_fuzz_roughness << "," << std::endl
            << "  coat_weight = " << m_coat_weight << "," << std::endl
            << "  coat_roughness = " << m_coat_roughness << "," << std::endl
            << "  coat_ior = " << m_coat_ior << "," << std::endl
            << "  thin_film_thickness = " << m_thin_film_thickness << "," << std::endl
            << "  thin_film_ior = " << m_thin_film_ior << "," << std::endl
            << "  thin_walled = " << m_thin_walled << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(OpenPBRSurface)

private:
    // Base layer
    ScalarFloat m_base_weight;
    ref<Texture> m_base_color;
    ref<Texture> m_base_metalness;

    // Specular
    ScalarFloat m_specular_weight;
    ref<Texture> m_specular_color;
    ref<Texture> m_specular_roughness;
    ScalarFloat m_specular_ior;
    ref<Texture> m_specular_anisotropy;

    // Transmission
    ref<Texture> m_transmission_weight;
    ref<Texture> m_transmission_color;
    ScalarFloat m_transmission_depth;

    // Subsurface
    ref<Texture> m_subsurface_weight;
    ref<Texture> m_subsurface_color;
    ScalarFloat m_subsurface_radius;

    // Fuzz/sheen
    ref<Texture> m_fuzz_weight;
    ref<Texture> m_fuzz_color;
    ref<Texture> m_fuzz_roughness;

    // Coat
    ref<Texture> m_coat_weight;
    ref<Texture> m_coat_color;
    ref<Texture> m_coat_roughness;
    ScalarFloat m_coat_ior;

    // Thin film
    ScalarFloat m_thin_film_thickness;
    ScalarFloat m_thin_film_ior;

    // Geometry
    bool m_thin_walled;

    // Feature flags
    bool m_has_metalness;
    bool m_has_transmission;
    bool m_has_coat;
    bool m_has_anisotropy;
    bool m_has_subsurface;
    bool m_has_fuzz;
    bool m_has_thin_film;

    MI_TRAVERSE_CB(Base, m_base_color, m_base_metalness,
                   m_specular_color, m_specular_roughness,
                   m_specular_anisotropy,
                   m_transmission_weight, m_transmission_color,
                   m_subsurface_weight, m_subsurface_color,
                   m_fuzz_weight, m_fuzz_color, m_fuzz_roughness,
                   m_coat_weight, m_coat_color, m_coat_roughness,
                   m_specular_ior, m_coat_ior,
                   m_thin_film_thickness, m_thin_film_ior,
                   m_transmission_depth, m_subsurface_radius)
};

MI_EXPORT_PLUGIN(OpenPBRSurface)
NAMESPACE_END(mitsuba)
