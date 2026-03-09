// SPDX-License-Identifier: MIT
// USD scene shape plugin for Mitsuba 3
//
// Loads an entire USD file and expands into one Mesh shape per GeomMesh prim,
// each with its own OpenPBR BSDF (base_color from material binding or displayColor).
// World transforms are computed by traversing the prim hierarchy.

#include <mitsuba/render/mesh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/profiler.h>

// tinyusdz
#include "tinyusdz.hh"
#include "usdGeom.hh"
#include "usdShade.hh"
#include "xform.hh"
#include "composition.hh"

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class USDScene final : public Shape<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Shape)
    MI_IMPORT_TYPES(BSDF)

    USDScene(const Properties &props) {
        // Note: NOT calling Shape constructor (like MergeShape) —
        // this is a pure container that expands into child shapes.
        auto fr = file_resolver();
        fs::path file_path = fr->resolve(props.get<std::string_view>("filename"));

        if (!fs::exists(file_path))
            Throw("USDScene: file not found: %s", file_path.string());

        ScopedPhase phase(ProfilerPhase::LoadGeometry);
        Timer timer;

        Log(Info, "Loading USD scene from \"%s\" ..", file_path.string());

        // --- Load USD as Layer, iteratively compose arcs ---
        tinyusdz::Layer root_layer;
        std::string warn, err;
        if (!tinyusdz::LoadLayerFromFile(
                fs::absolute(file_path).string(), &root_layer, &warn, &err))
            Throw("USDScene: load failed: %s", err.c_str());

        std::string base_dir = fs::absolute(file_path).parent_path().string();
        tinyusdz::AssetResolutionResolver resolver;
        resolver.set_current_working_path(base_dir);
        resolver.set_search_paths({base_dir});

        // Iteratively resolve composition arcs (LIVRPS order)
        // until no unresolved arcs remain
        tinyusdz::Layer src_layer = root_layer;
        constexpr int kMaxIter = 32;
        for (int iter = 0; iter < kMaxIter; ++iter) {
            bool has_unresolved = false;

            if (src_layer.check_unresolved_references()) {
                has_unresolved = true;
                tinyusdz::Layer comp;
                if (!tinyusdz::CompositeReferences(resolver, src_layer,
                                                    &comp, &warn, &err))
                    Throw("USDScene: reference composition failed: %s",
                          err.c_str());
                src_layer = std::move(comp);
            }

            if (src_layer.check_unresolved_payload()) {
                has_unresolved = true;
                tinyusdz::Layer comp;
                if (!tinyusdz::CompositePayload(resolver, src_layer,
                                                 &comp, &warn, &err))
                    Throw("USDScene: payload composition failed: %s",
                          err.c_str());
                src_layer = std::move(comp);
            }

            if (src_layer.check_unresolved_variant()) {
                has_unresolved = true;
                tinyusdz::Layer comp;
                if (!tinyusdz::CompositeVariant(src_layer, &comp, &warn, &err))
                    Throw("USDScene: variant composition failed: %s",
                          err.c_str());
                src_layer = std::move(comp);
            }

            if (!has_unresolved)
                break;
        }

        // Convert to Stage
        tinyusdz::Stage stage;
        if (!tinyusdz::LayerToStage(src_layer, &stage, &warn, &err))
            Throw("USDScene: LayerToStage failed: %s", err.c_str());
        stage.compute_absolute_prim_path_and_assign_prim_id();

        // --- Traverse prim tree, create one usdmesh shape per GeomMesh ---
        size_t mesh_count = 0;
        auto identity = tinyusdz::value::matrix4d::identity();

        for (const auto &root_prim : stage.root_prims())
            traverse(root_prim, identity, file_path.string(), stage,
                     mesh_count);

        Log(Info, "USDScene \"%s\": %zu meshes (took %s)",
            file_path.filename().string(), mesh_count,
            util::time_string((float) timer.value()));
    }

    std::vector<ref<Object>> expand() const override {
        return m_shapes;
    }

    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    MI_DECLARE_CLASS(USDScene)

private:
    using ScalarTransform4f = typename Shape<Float, Spectrum>::ScalarTransform4f;

    /// Convert tinyusdz matrix4d (row-major) to Mitsuba ScalarTransform4f (column-major)
    static ScalarTransform4f to_mi_transform(const tinyusdz::value::matrix4d &m) {
        // USD row-vector convention: v' = v * M
        // Mitsuba column-vector: v' = M * v
        // => transpose
        using Matrix4f = dr::Matrix<float, 4>;
        Matrix4f mat;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                mat(i, j) = (float) m.m[j][i]; // transpose
        return ScalarTransform4f(mat);
    }

    /// Get the local xform matrix for a prim, if it's xformable
    static tinyusdz::value::matrix4d get_local_xform(const tinyusdz::Prim &prim) {
        // Try GeomMesh
        if (auto *mesh = prim.as<tinyusdz::GeomMesh>()) {
            auto result = mesh->GetLocalMatrix();
            if (result) return result.value();
        }
        // Try Xform
        if (auto *xform = prim.as<tinyusdz::Xform>()) {
            auto result = xform->GetLocalMatrix();
            if (result) return result.value();
        }
        return tinyusdz::value::matrix4d::identity();
    }

    /// Get material color for a GeomMesh prim.
    /// Priority: material:binding → displayColor → default grey.
    static ScalarColor3f get_material_color(
        const tinyusdz::GeomMesh *mesh,
        const tinyusdz::Prim &prim,
        const tinyusdz::Stage &stage)
    {
        // 1. Try material:binding → UsdPreviewSurface.diffuseColor
        if (mesh->has_materialBinding()) {
            auto &rel = mesh->materialBinding.value();
            auto target = rel.targetPath;
            if (target.is_valid()) {
                const tinyusdz::Prim *mat_prim = nullptr;
                std::string err;
                if (stage.find_prim_at_path(
                        tinyusdz::Path(target.prim_part(), ""),
                        mat_prim, &err) && mat_prim) {
                    auto *material = mat_prim->as<tinyusdz::Material>();
                    if (material && material->surface.authored()) {
                        auto conns = material->surface.get_connections();
                        if (!conns.empty()) {
                            const tinyusdz::Prim *shader_prim = nullptr;
                            if (stage.find_prim_at_path(
                                    tinyusdz::Path(conns[0].prim_part(), ""),
                                    shader_prim, &err) && shader_prim) {
                                auto *shader = shader_prim->as<tinyusdz::Shader>();
                                if (shader) {
                                    auto *ps = shader->value.as<tinyusdz::UsdPreviewSurface>();
                                    if (ps) {
                                        tinyusdz::value::color3f c;
                                        if (ps->diffuseColor.get_value().get_scalar(&c))
                                            return ScalarColor3f(c[0], c[1], c[2]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Try displayColor primvar (single value or array)
        tinyusdz::value::color3f col;
        if (mesh->get_displayColor(&col))
            return ScalarColor3f(col[0], col[1], col[2]);

        // 3. Try displayColors array (use first color)
        auto colors = mesh->get_displayColors();
        if (!colors.empty())
            return ScalarColor3f(colors[0][0], colors[0][1], colors[0][2]);

        // 4. Default grey
        return ScalarColor3f(0.5f, 0.5f, 0.5f);
    }

    /// Recursively traverse prims, creating shapes for each GeomMesh
    void traverse(const tinyusdz::Prim &prim,
                  const tinyusdz::value::matrix4d &parent_world,
                  const std::string &filename,
                  const tinyusdz::Stage &stage,
                  size_t &mesh_count) {
        // Compute this prim's world matrix
        auto local = get_local_xform(prim);
        // Row-major: world = local * parent
        auto world = local * parent_world;

        // If this prim is a GeomMesh, create a shape
        if (auto *mesh = prim.as<tinyusdz::GeomMesh>()) {
            auto points = mesh->get_points();
            if (!points.empty()) {
                auto color = get_material_color(mesh, prim, stage);
                std::string prim_path = prim.absolute_path().full_path_name();

                // Create OpenPBR BSDF with material color
                Properties bsdf_props("openpbr");
                bsdf_props.set("base_color", color);
                auto bsdf = PluginManager::instance()
                    ->create_object<BSDF>(bsdf_props);

                // Create usdmesh shape
                Properties mesh_props("usdmesh");
                mesh_props.set("filename", filename);
                mesh_props.set("prim_path", prim_path);
                mesh_props.set("to_world", to_mi_transform(world));
                mesh_props.set("bsdf", ref<Object>(bsdf.get()));

                auto shape = PluginManager::instance()
                    ->create_object<Shape>(mesh_props);
                m_shapes.push_back(ref<Object>(shape.get()));

                mesh_count++;
            }
        }

        // Recurse into children
        for (const auto &child : prim.children())
            traverse(child, world, filename, stage, mesh_count);
    }

    std::vector<ref<Object>> m_shapes;
};

MI_EXPORT_PLUGIN(USDScene)
NAMESPACE_END(mitsuba)
