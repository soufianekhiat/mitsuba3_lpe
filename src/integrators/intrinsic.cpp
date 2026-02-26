/**
 * Intrinsic Properties Integrator
 *
 * A first-hit integrator that extracts material and geometric properties
 * as AOV channels for intrinsic image decomposition.
 *
 * Output AOV channels (12 total):
 *   lin_z.T           Linear depth (distance from camera)
 *   normal.X/Y/Z      Camera-space geometric normal
 *   diff_alb.R/G/B    Diffuse albedo (eval_diffuse_reflectance)
 *   spec_alb.R/G/B    Specular albedo (estimated from BSDF attributes)
 *   rough.T            Surface roughness [0..1]
 *   shadow.T           Direct light visibility (1=lit, 0=shadowed)
 *
 * Specular albedo estimation strategy:
 *   - Query "specular_reflectance" attribute if available (rough* BSDFs)
 *   - Otherwise estimate Fresnel at normal incidence via "eta" attribute
 *   - Pure diffuse BSDFs get spec_alb = 0
 *
 * Roughness estimation strategy:
 *   - Query "alpha" attribute (rough* BSDFs)
 *   - Or "roughness" attribute (principled BSDF)
 *   - Fallback: diffuse=1, delta=0, other=0.5
 *
 * Shadow: single-bounce binary occlusion (deterministic, no spp averaging).
 *   Uses fixed sample to pick the dominant emitter direction, tests visibility.
 *   Result is pure black (0) or white (1), never grayscale.
 */

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class IntrinsicIntegrator final : public SamplingIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SamplingIntegrator)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, BSDFPtr, ShapePtr, Sensor)

    IntrinsicIntegrator(const Properties &props) : Base(props) { }

    static constexpr size_t AOV_COUNT = 12;

    std::vector<std::string> aov_names() const override {
        return {
            "lin_z.T",
            "normal.X",   "normal.Y",   "normal.Z",
            "diff_alb.R", "diff_alb.G", "diff_alb.B",
            "spec_alb.R", "spec_alb.G", "spec_alb.B",
            "rough.T",
            "shadow.T"
        };
    }

    std::pair<Spectrum, Mask> sample(
        const Scene *scene,
        Sampler *sampler,
        const RayDifferential3f &ray,
        const Medium * /* medium */,
        Float *aovs,
        Mask active
    ) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        // Ray intersection
        SurfaceInteraction3f si =
            scene->ray_intersect(ray, (uint32_t) RayFlags::All,
                                 /* coherent */ true, /* reorder */ false,
                                 0, 0, active);

        Mask valid = active && si.is_valid();

        // Initialize AOVs to zero
        if (aovs) {
            for (size_t i = 0; i < AOV_COUNT; ++i)
                aovs[i] = 0.f;
        }

        if (dr::none_or<false>(valid) || !aovs)
            return { 0.f, valid };

        // ===================== Linear Depth =====================
        aovs[0] = dr::select(valid, si.t, 0.f);

        // ===================== BSDF =====================
        BSDFPtr bsdf = si.bsdf(ray);

        // ===================== Camera-space Geometric Normal =====================
        {
            const auto &sensors = scene->sensors();
            if (!sensors.empty()) {
                // Camera-to-world transform (returns value, not pointer)
                auto c2w = sensors[0]->world_transform();

                // Camera basis vectors in world space (use operator*)
                Vector3f cx = c2w * Vector3f(1, 0, 0);
                Vector3f cy = c2w * Vector3f(0, 1, 0);
                Vector3f cz = c2w * Vector3f(0, 0, 1);

                // Project geometric normal onto camera axes
                Normal3f n = si.n;
                aovs[1] = dr::select(valid, dr::dot(Vector3f(n), cx), 0.f);
                aovs[2] = dr::select(valid, dr::dot(Vector3f(n), cy), 0.f);
                aovs[3] = dr::select(valid, dr::dot(Vector3f(n), cz), 0.f);
            }
        }

        // ===================== Spectrum -> Color3f helper =====================
        auto spec_to_rgb = [&](const Spectrum &spec, Mask m) -> Color3f {
            DRJIT_MARK_USED(m);
            UnpolarizedSpectrum spec_u = unpolarized_spectrum(spec);
            if constexpr (is_monochromatic_v<Spectrum>) {
                Float v = spec_u.x();
                return Color3f(v, v, v);
            } else if constexpr (is_rgb_v<Spectrum>) {
                return Color3f(spec_u.x(), spec_u.y(), spec_u.z());
            } else {
                static_assert(is_spectral_v<Spectrum>);
                auto pdf = pdf_rgb_spectrum(ray.wavelengths);
                spec_u *= dr::select(pdf != 0.f, dr::rcp(pdf), 0.f);
                return spectrum_to_srgb(spec_u, ray.wavelengths, m);
            }
        };

        // ===================== Diffuse Albedo =====================
        {
            Color3f rgb(0.f);
            Spectrum spec = bsdf->eval_diffuse_reflectance(si, valid);
            dr::masked(rgb, valid) = spec_to_rgb(spec, valid);
            aovs[4] = rgb.r();
            aovs[5] = rgb.g();
            aovs[6] = rgb.b();
        }

        // ===================== Specular Albedo & Roughness =====================
        // In scalar mode, we can query BSDF attributes via traverse().
        // In JIT mode, has_attribute() returns a drjit Mask that can't be
        // used in if-conditionals, so we use flag-based heuristics instead.

        if constexpr (!dr::is_jit_v<Float>) {
            // --- Scalar path: query BSDF attributes directly ---

            // Specular albedo
            {
                Color3f rgb(0.f);
                bool has_spec_flag =
                    has_flag(bsdf->flags(), BSDFFlags::Glossy) ||
                    has_flag(bsdf->flags(), BSDFFlags::Delta);

                if (valid && has_spec_flag) {
                    if (bsdf->has_attribute("specular_reflectance", valid)) {
                        rgb = bsdf->eval_attribute_3(
                            "specular_reflectance", si, valid);
                    } else {
                        // Fresnel F0 = ((eta-1)/(eta+1))^2
                        Float eta_val = 1.5f;
                        if (bsdf->has_attribute("eta", valid))
                            eta_val = bsdf->eval_attribute_1(
                                "eta", si, valid);
                        Float F0 = dr::square(
                            (eta_val - 1.f) / (eta_val + 1.f));
                        rgb = Color3f(F0);
                    }
                }
                aovs[7] = rgb.r();
                aovs[8] = rgb.g();
                aovs[9] = rgb.b();
            }

            // Roughness
            {
                Float rough = 0.f;
                if (valid) {
                    if (bsdf->has_attribute("alpha", valid)) {
                        rough = bsdf->eval_attribute_1("alpha", si, valid);
                    } else if (bsdf->has_attribute("roughness", valid)) {
                        rough = bsdf->eval_attribute_1(
                            "roughness", si, valid);
                    } else {
                        // Heuristic
                        bool is_diff =
                            has_flag(bsdf->flags(), BSDFFlags::Diffuse) &&
                            !has_flag(bsdf->flags(), BSDFFlags::Glossy) &&
                            !has_flag(bsdf->flags(), BSDFFlags::Delta);
                        bool is_delta =
                            has_flag(bsdf->flags(), BSDFFlags::Delta);
                        rough = is_diff ? 1.f : (is_delta ? 0.f : 0.5f);
                    }
                }
                aovs[10] = rough;
            }
        } else {
            // --- JIT path: flag-based heuristics ---

            // Specular albedo: use Fresnel F0 estimate for specular BSDFs
            {
                Color3f rgb(0.f);
                Bool has_spec =
                    has_flag(bsdf->flags(), BSDFFlags::Glossy) ||
                    has_flag(bsdf->flags(), BSDFFlags::Delta);
                // Default F0 for glass/plastic (eta ~1.5)
                Float F0 = dr::square((1.5f - 1.f) / (1.5f + 1.f));
                dr::masked(rgb, valid && has_spec) = Color3f(F0);
                aovs[7] = rgb.r();
                aovs[8] = rgb.g();
                aovs[9] = rgb.b();
            }

            // Roughness: heuristic from flags
            {
                Float rough = 0.5f;
                Bool is_diff =
                    has_flag(bsdf->flags(), BSDFFlags::Diffuse) &&
                    !has_flag(bsdf->flags(), BSDFFlags::Glossy) &&
                    !has_flag(bsdf->flags(), BSDFFlags::Delta);
                Bool is_delta =
                    has_flag(bsdf->flags(), BSDFFlags::Delta);
                rough = dr::select(is_diff, 1.f,
                         dr::select(is_delta, 0.f, 0.5f));
                aovs[10] = dr::select(valid, rough, 0.f);
            }
        }

        // ===================== Shadow (single-bounce binary occlusion) =====================
        // Deterministic: uses fixed sample (0.5, 0.5) so every spp produces
        // the same result → pure binary (black/white, no grayscale).
        // Tests visibility toward the importance-sampled emitter direction
        // (including environment maps). 1 = lit, 0 = occluded.
        {
            Float vis = 0.f;

            if (dr::any_or<true>(valid)) {
                // Fixed sample ensures identical result across all spp
                Point2f fixed_sample(0.5f, 0.5f);
                auto [ds, em_weight] = scene->sample_emitter_direction(
                    si, fixed_sample,
                    /* test_visibility */ true, valid);
                // pdf > 0 means the emitter is visible (not occluded)
                dr::masked(vis, valid) =
                    dr::select(ds.pdf > 0.f, 1.f, 0.f);
            }
            aovs[11] = vis;
        }

        return { 0.f, valid };
    }

    std::string to_string() const override {
        return "IntrinsicIntegrator[]";
    }

    MI_DECLARE_CLASS(IntrinsicIntegrator)
};

MI_EXPORT_PLUGIN(IntrinsicIntegrator)
NAMESPACE_END(mitsuba)
