/**
 * Light Path Expression (LPE) Integrator
 *
 * A path tracer that decomposes the rendered image into multiple AOV channels
 * based on Light Path Expressions. Each bounce is classified as:
 *   D = Diffuse  (BSDFFlags::Diffuse)
 *   G = Glossy   (BSDFFlags::Glossy)
 *   S = Specular  (BSDFFlags::Delta)
 *   T = Transmission (BSDFFlags::*Transmission, excluding Null)
 *
 * Standard LPE notation:
 *   L  = Light source
 *   E  = Eye / Camera
 *   .  = Any interaction
 *   *  = Zero or more
 *   +  = One or more
 *
 * Output channels:
 *   beauty        L.*E          Full render (same as path integrator)
 *   direct_diff   LDE           Direct diffuse (first-bounce diffuse)
 *   direct_spec   LSE           Direct specular (first-bounce specular)
 *   indirect_diff L.+DE         Indirect diffuse
 *   indirect_spec L.+SE         Indirect specular
 *   emissive      LE            Directly visible emitters
 *   transmission  L.*T.*E       Light through transmissive surfaces
 */

#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class LPEIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    LPEIntegrator(const Properties &props) : Base(props) { }

    /// LPE AOV channels: 6 passes x RGB = 18 floats
    /// direct_diff, direct_spec, indirect_diff, indirect_spec, emissive, transmission
    static constexpr size_t NUM_LPE_PASSES = 6;
    static constexpr size_t LPE_AOV_COUNT  = NUM_LPE_PASSES * 3;

    enum LPEPass {
        DirectDiffuse   = 0,
        DirectSpecular  = 1,
        IndirectDiffuse = 2,
        IndirectSpecular= 3,
        Emissive        = 4,
        Transmission    = 5
    };

    std::vector<std::string> aov_names() const override {
        return {
            "direct_diff.R",  "direct_diff.G",  "direct_diff.B",
            "direct_spec.R",  "direct_spec.G",  "direct_spec.B",
            "indirect_diff.R","indirect_diff.G", "indirect_diff.B",
            "indirect_spec.R","indirect_spec.G", "indirect_spec.B",
            "emissive.R",     "emissive.G",      "emissive.B",
            "transmission.R", "transmission.G",  "transmission.B"
        };
    }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float *aovs,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        PreliminaryIntersection3f pi  = dr::zeros<PreliminaryIntersection3f>();
        UInt32 depth                  = 0;

        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        // Variables caching information from the previous bounce
        Interaction3f prev_si               = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf         = 1.f;
        Bool          prev_bsdf_delta       = true;
        Bool          prev_bsdf_transmission = false;
        BSDFContext   bsdf_ctx;

        // LPE accumulation channels
        Spectrum lpe_direct_diff   = 0.f;
        Spectrum lpe_direct_spec   = 0.f;
        Spectrum lpe_indirect_diff = 0.f;
        Spectrum lpe_indirect_spec = 0.f;
        Spectrum lpe_emissive      = 0.f;
        Spectrum lpe_transmission  = 0.f;

        struct LoopState {
            Ray3f ray;
            PreliminaryIntersection3f pi;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            UInt32 depth;
            Mask valid_ray;
            Interaction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta;
            Bool prev_bsdf_transmission;
            Bool active;
            Sampler* sampler;
            // LPE channels
            Spectrum lpe_direct_diff;
            Spectrum lpe_direct_spec;
            Spectrum lpe_indirect_diff;
            Spectrum lpe_indirect_spec;
            Spectrum lpe_emissive;
            Spectrum lpe_transmission;

            DRJIT_STRUCT(LoopState, ray, pi, throughput, result, eta, depth, \
                valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                prev_bsdf_transmission, active, sampler,
                lpe_direct_diff, lpe_direct_spec,
                lpe_indirect_diff, lpe_indirect_spec, lpe_emissive,
                lpe_transmission)
        } ls = {
            ray, pi, throughput, result, eta, depth,
            valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
            prev_bsdf_transmission, active, sampler,
            lpe_direct_diff, lpe_direct_spec,
            lpe_indirect_diff, lpe_indirect_spec, lpe_emissive,
            lpe_transmission
        };

        // First bounce - coherent
        ls.pi = scene->ray_intersect_preliminary(ls.ray, true, false, 0, 0, ls.active);

        // ---------------------- Hide area emitters ----------------------

        if (m_hide_emitters && dr::any_or<true>(ls.depth == 0u)) {
            Mask skip_emitters = ls.pi.is_valid() &&
                                 (ls.pi.shape->emitter() != nullptr) &&
                                 ls.active;

            if (dr::any_or<true>(skip_emitters)) {
                SurfaceInteraction3f si = ls.pi.compute_surface_interaction(
                    ls.ray, +RayFlags::Minimal, skip_emitters);
                Ray3f ray2 = si.spawn_ray(ls.ray.d);
                PreliminaryIntersection3f pi_after_skip =
                    Base::skip_area_emitters(scene, ray2, true, skip_emitters);
                dr::masked(ls.pi, skip_emitters) = pi_after_skip;
            }
        }

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return ls.active; },
            [this, scene, bsdf_ctx](LoopState& ls) {

            SurfaceInteraction3f si =
                ls.pi.compute_surface_interaction(ls.ray, +RayFlags::All);

            // ---------------------- Direct emission ----------------------
            // This handles hitting an emitter via BSDF sampling

            if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                DirectionSample3f ds(scene, si, ls.prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!ls.prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(ls.prev_si, ds,
                                                          !ls.prev_bsdf_delta);

                Float mis_bsdf = mis_weight(ls.prev_bsdf_pdf, em_pdf);
                Spectrum emission_contrib =
                    ds.emitter->eval(si, ls.prev_bsdf_pdf > 0.f) * mis_bsdf;
                Spectrum weighted = ls.throughput * emission_contrib;

                // Beauty accumulation
                ls.result = spec_fma(ls.throughput, emission_contrib, ls.result);

                // LPE classification of BSDF-sampled emitter hits:
                // depth==0 means directly visible emitter (LE)
                // depth>=1 with transmission last bounce -> transmission
                // depth==1 with reflection last bounce -> direct diff/spec
                // depth>=2 with reflection last bounce -> indirect diff/spec
                Mask is_emissive_vis = (ls.depth == 0u);
                ls.lpe_emissive[is_emissive_vis] += weighted;

                Mask depth_ge1 = (ls.depth >= 1u);

                // Transmission: last bounce was a transmissive event
                ls.lpe_transmission[depth_ge1 && ls.prev_bsdf_transmission] += weighted;

                // Reflection passes (non-transmission)
                Mask depth_1_refl   = (ls.depth == 1u)  && !ls.prev_bsdf_transmission;
                Mask depth_ge2_refl = (ls.depth >= 2u) && !ls.prev_bsdf_transmission;

                // Direct: single reflection bounce
                ls.lpe_direct_diff[depth_1_refl && !ls.prev_bsdf_delta]  += weighted;
                ls.lpe_direct_spec[depth_1_refl && ls.prev_bsdf_delta]   += weighted;

                // Indirect: 2+ bounces, last was reflection
                ls.lpe_indirect_diff[depth_ge2_refl && !ls.prev_bsdf_delta] += weighted;
                ls.lpe_indirect_spec[depth_ge2_refl && ls.prev_bsdf_delta]  += weighted;
            }

            // Continue tracing?
            Bool active_next = (ls.depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next)) {
                ls.active = active_next;
                ls.valid_ray |= (si.emitter(scene) != nullptr) && !m_hide_emitters;
                return;
            }

            BSDFPtr bsdf = si.bsdf(ls.ray);

            // ---------------------- Emitter sampling ----------------------

            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, ls.sampler->next_2d(), true, active_em);
                active_em &= (ds.pdf != 0.f);

                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(ds.pdf != 0, em_val / ds.pdf, 0);
                }

                wo = si.to_local(ds.d);
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            Float sample_1 = ls.sampler->next_1d();
            Point2f sample_2 = ls.sampler->next_2d();

            auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight]
                = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);

            // --------------- Emitter sampling contribution ----------------

            if (dr::any_or<true>(active_em)) {
                Spectrum bsdf_val_w = si.to_world_mueller(bsdf_val, -wo, si.wi);
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                Spectrum nee_contrib = bsdf_val_w * em_weight * mis_em;
                Spectrum weighted_nee = ls.throughput * nee_contrib;

                // Beauty accumulation
                ls.result[active_em] = spec_fma(
                    ls.throughput, nee_contrib, ls.result);

                // LPE for emitter sampling (NEE):
                // The NEE direction wo is in local frame. If wo.z() < 0,
                // the light is on the back side -> transmission evaluation,
                // but ONLY if the BSDF actually supports transmission.
                // (twosided BSDFs respond to wo.z < 0 via normal flipping,
                //  not real transmission -- exclude those)
                Bool bsdf_has_real_transmission =
                    has_flag(bsdf->flags(), BSDFFlags::DiffuseTransmission) ||
                    has_flag(bsdf->flags(), BSDFFlags::GlossyTransmission) ||
                    has_flag(bsdf->flags(), BSDFFlags::DeltaTransmission) ||
                    has_flag(bsdf->flags(), BSDFFlags::Delta1DTransmission);
                Bool nee_is_transmission =
                    bsdf_has_real_transmission && (Frame3f::cos_theta(wo) < 0.f);

                // Transmission: NEE through transmissive surface
                Mask nee_trans = active_em && nee_is_transmission;
                ls.lpe_transmission[nee_trans] += weighted_nee;

                // Reflection: classify as diffuse/specular
                Mask nee_refl = active_em && !nee_is_transmission;
                Bool nee_has_diffuse = has_flag(bsdf->flags(), BSDFFlags::DiffuseReflection) ||
                                       has_flag(bsdf->flags(), BSDFFlags::GlossyReflection);

                Mask nee_depth_0   = nee_refl && (ls.depth == 0u);
                Mask nee_depth_ge1 = nee_refl && (ls.depth >= 1u);

                // Direct (first surface, depth==0)
                ls.lpe_direct_diff[nee_depth_0 && nee_has_diffuse]   += weighted_nee;
                ls.lpe_direct_spec[nee_depth_0 && !nee_has_diffuse]  += weighted_nee;

                // Indirect (depth >= 1)
                ls.lpe_indirect_diff[nee_depth_ge1 && nee_has_diffuse]  += weighted_nee;
                ls.lpe_indirect_spec[nee_depth_ge1 && !nee_has_diffuse] += weighted_nee;
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ls.ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            if (dr::grad_enabled(ls.ray)) {
                ls.ray = dr::detach<true>(ls.ray);
                Vector3f wo_2 = si.to_local(ls.ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = bsdf->eval_pdf(bsdf_ctx, si, wo_2, ls.active);
                bsdf_weight[bsdf_pdf_2 > 0.f] = bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables ------

            ls.throughput *= bsdf_weight;
            ls.eta *= bsdf_sample.eta;
            ls.valid_ray |= ls.active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            ls.prev_si = Interaction3f(si);
            ls.prev_bsdf_pdf = bsdf_sample.pdf;
            ls.prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);
            ls.prev_bsdf_transmission =
                has_flag(bsdf_sample.sampled_type, BSDFFlags::DiffuseTransmission) ||
                has_flag(bsdf_sample.sampled_type, BSDFFlags::GlossyTransmission) ||
                has_flag(bsdf_sample.sampled_type, BSDFFlags::DeltaTransmission) ||
                has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta1DTransmission);

            // -------------------- Stopping criterion ---------------------

            dr::masked(ls.depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);
            Mask rr_active = ls.depth >= m_rr_depth,
                 rr_continue = ls.sampler->next_1d() < rr_prob;

            ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            ls.active = active_next && (!rr_active || rr_continue) &&
                        (throughput_max != 0.f);

            ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                     false,
                                                     jit_flag(JitFlag::LoopRecord),
                                                     0, 0,
                                                     ls.active);
        });

        // ---------------------- Write AOVs ----------------------

        if (aovs != nullptr) {
            auto write_rgb = [&](Spectrum spec, Float *out, Mask valid) {
                spec = dr::select(valid, spec, 0.f);
                UnpolarizedSpectrum spec_u = unpolarized_spectrum(spec);
                if constexpr (is_monochromatic_v<Spectrum>) {
                    Float v = spec_u.x();
                    *out++ = v; *out++ = v; *out++ = v;
                } else if constexpr (is_rgb_v<Spectrum>) {
                    *out++ = spec_u.x(); *out++ = spec_u.y(); *out++ = spec_u.z();
                } else {
                    static_assert(is_spectral_v<Spectrum>);
                    auto pdf = pdf_rgb_spectrum(ray_.wavelengths);
                    spec_u *= dr::select(pdf != 0.f, dr::rcp(pdf), 0.f);
                    Color3f rgb = spectrum_to_srgb(spec_u, ray_.wavelengths, valid);
                    *out++ = rgb.x(); *out++ = rgb.y(); *out++ = rgb.z();
                }
            };

            write_rgb(ls.lpe_direct_diff,   aovs + DirectDiffuse   * 3, ls.valid_ray);
            write_rgb(ls.lpe_direct_spec,   aovs + DirectSpecular  * 3, ls.valid_ray);
            write_rgb(ls.lpe_indirect_diff, aovs + IndirectDiffuse * 3, ls.valid_ray);
            write_rgb(ls.lpe_indirect_spec, aovs + IndirectSpecular* 3, ls.valid_ray);
            write_rgb(ls.lpe_emissive,      aovs + Emissive        * 3, ls.valid_ray);
            write_rgb(ls.lpe_transmission,  aovs + Transmission    * 3, ls.valid_ray);
        }

        return {
            dr::select(ls.valid_ray, ls.result, 0.f),
            ls.valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("LPEIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS(LPEIntegrator)
};

MI_EXPORT_PLUGIN(LPEIntegrator)
NAMESPACE_END(mitsuba)
