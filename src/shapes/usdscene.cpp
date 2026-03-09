// SPDX-License-Identifier: MIT
// USD scene shape plugin for Mitsuba 3
//
// Loads an entire USD file and expands into one Mesh shape per GeomMesh prim,
// each with its own OpenPBR BSDF (from MaterialX open_pbr_surface, material
// binding, displayColor, or default grey).
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

#include <array>
#include <fstream>
#include <sstream>
#include <unordered_map>

NAMESPACE_BEGIN(mitsuba)

// ---------------------------------------------------------------------------
// Simple MaterialX open_pbr_surface constant parameter extractor
// ---------------------------------------------------------------------------

/// Parsed constant OpenPBR parameters from a .mtlx file
struct MtlxOpenPBRParams {
    bool valid = false;

    // base
    float base_weight = 1.0f;
    std::array<float,3> base_color = {0.8f, 0.8f, 0.8f};
    bool has_base_color = false;
    float base_metalness = 0.0f;
    bool has_metalness = false;

    // specular
    float specular_weight = 1.0f;
    bool has_specular_weight = false;
    std::array<float,3> specular_color = {1.0f, 1.0f, 1.0f};
    bool has_specular_color = false;
    float specular_roughness = 0.3f;
    bool has_specular_roughness = false;
    float specular_ior = 1.5f;
    bool has_specular_ior = false;

    // transmission
    float transmission_weight = 0.0f;
    bool has_transmission = false;

    // coat
    float coat_weight = 0.0f;
    bool has_coat = false;
    std::array<float,3> coat_color = {1.0f, 1.0f, 1.0f};
    bool has_coat_color = false;
    float coat_roughness = 0.0f;
    bool has_coat_roughness = false;
    float coat_ior = 1.5f;
    bool has_coat_ior = false;
};

/// Read entire file to string
static std::string read_file_to_string(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

/// Trim whitespace
static std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    return s.substr(a, b - a + 1);
}

/// Extract attribute value from an XML element string: name="value"
static std::string xml_attr(const std::string &elem, const std::string &attr) {
    std::string needle = attr + "=\"";
    auto pos = elem.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    auto end = elem.find('"', pos);
    if (end == std::string::npos) return {};
    return elem.substr(pos, end - pos);
}

/// Parse a float from string
static float parse_float(const std::string &s, float fallback = 0.0f) {
    try { return std::stof(s); }
    catch (...) { return fallback; }
}

/// Parse "r, g, b" color3 string
static std::array<float,3> parse_color3(const std::string &s) {
    std::array<float,3> out = {0.5f, 0.5f, 0.5f};
    std::string clean = s;
    for (char &c : clean) if (c == ',') c = ' ';
    std::istringstream iss(clean);
    iss >> out[0] >> out[1] >> out[2];
    return out;
}

/// Parse a .mtlx file and extract constant open_pbr_surface inputs
static MtlxOpenPBRParams parse_mtlx_openpbr(const std::string &mtlx_path) {
    MtlxOpenPBRParams params;
    std::string xml = read_file_to_string(mtlx_path);
    if (xml.empty()) return params;

    // Find the open_pbr_surface element
    auto openpbr_pos = xml.find("<open_pbr_surface");
    if (openpbr_pos == std::string::npos) return params;

    // Find the closing tag
    auto close_pos = xml.find("</open_pbr_surface>", openpbr_pos);
    if (close_pos == std::string::npos) {
        // Self-closing?
        close_pos = xml.find("/>", openpbr_pos);
        if (close_pos == std::string::npos) return params;
    }

    params.valid = true;
    std::string block = xml.substr(openpbr_pos, close_pos - openpbr_pos + 20);

    // Extract all <input> elements within the block
    size_t search_pos = 0;
    while (true) {
        auto inp_pos = block.find("<input", search_pos);
        if (inp_pos == std::string::npos) break;

        auto inp_end = block.find(">", inp_pos);
        if (inp_end == std::string::npos) break;
        // Include the > or />
        std::string inp = block.substr(inp_pos, inp_end - inp_pos + 1);

        std::string name = xml_attr(inp, "name");
        std::string type = xml_attr(inp, "type");
        std::string value = xml_attr(inp, "value");
        std::string nodename = xml_attr(inp, "nodename");

        // Only process constant values (have "value", no "nodename")
        if (!value.empty() && nodename.empty() && !name.empty()) {
            if (name == "base_weight") {
                params.base_weight = parse_float(value, 1.0f);
            } else if (name == "base_color" && type == "color3") {
                params.base_color = parse_color3(value);
                params.has_base_color = true;
            } else if (name == "base_metalness") {
                params.base_metalness = parse_float(value, 0.0f);
                params.has_metalness = true;
            } else if (name == "specular_weight") {
                params.specular_weight = parse_float(value, 1.0f);
                params.has_specular_weight = true;
            } else if (name == "specular_color" && type == "color3") {
                params.specular_color = parse_color3(value);
                params.has_specular_color = true;
            } else if (name == "specular_roughness") {
                params.specular_roughness = parse_float(value, 0.3f);
                params.has_specular_roughness = true;
            } else if (name == "specular_ior") {
                params.specular_ior = parse_float(value, 1.5f);
                params.has_specular_ior = true;
            } else if (name == "transmission_weight") {
                params.transmission_weight = parse_float(value, 0.0f);
                params.has_transmission = true;
            } else if (name == "coat_weight") {
                params.coat_weight = parse_float(value, 0.0f);
                params.has_coat = true;
            } else if (name == "coat_color" && type == "color3") {
                params.coat_color = parse_color3(value);
                params.has_coat_color = true;
            } else if (name == "coat_roughness") {
                params.coat_roughness = parse_float(value, 0.0f);
                params.has_coat_roughness = true;
            } else if (name == "coat_ior") {
                params.coat_ior = parse_float(value, 1.5f);
                params.has_coat_ior = true;
            }
        }

        search_pos = inp_end + 1;
    }

    return params;
}

// ---------------------------------------------------------------------------

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
        m_base_dir = base_dir;

        tinyusdz::AssetResolutionResolver resolver;
        resolver.set_current_working_path(base_dir);
        // Add base_dir + all immediate subdirs to search paths so that
        // recursive sublayer composition can find assets after CWP changes.
        std::vector<std::string> search_paths = {base_dir};
        for (const auto &sub : {"cameras", "materials", "assets", "lights",
                                 "textures", "geometry"}) {
            std::string sub_dir = base_dir + "/" + sub;
            if (fs::exists(sub_dir))
                search_paths.push_back(sub_dir);
        }
        resolver.set_search_paths(search_paths);

        // LIVRPS composition order:
        // 1. SubLayers (Local) — process each sublayer individually
        //    using CompositeSublayers with a fresh resolver each time,
        //    because tinyusdz's LoadAsset permanently modifies the resolver.
        tinyusdz::Layer src_layer = root_layer;
        if (!src_layer.metas().subLayers.empty()) {
            Log(Info, "USDScene: compositing %zu sublayers...",
                src_layer.metas().subLayers.size());

            // Process sublayers in REVERSE order: weakest (last) first, so that
            // base definitions (Mesh types, geometry) are established before
            // stronger opinions (material bindings, cameras) overlay on top.
            // In USD, subLayers[0] is strongest, subLayers[N-1] is weakest.
            size_t n_sublayers = src_layer.metas().subLayers.size();
            for (size_t ri = 0; ri < n_sublayers; ++ri) {
                size_t i = n_sublayers - 1 - ri;
                const auto &sl = src_layer.metas().subLayers[i];
                std::string sl_path = sl.assetPath.GetAssetPath();
                Log(Info, "USDScene: sublayer[%zu] = \"%s\"", i, sl_path.c_str());
                // Resolve relative to base_dir
                std::string sl_abs = base_dir + "/" + sl_path;
                // Normalize ./
                auto pos = sl_abs.find("/./");
                while (pos != std::string::npos) {
                    sl_abs.erase(pos, 2);
                    pos = sl_abs.find("/./");
                }

                if (!fs::exists(sl_abs)) {
                    Log(Warn, "USDScene: sublayer not found, skipping: %s",
                        sl_path.c_str());
                    continue;
                }

                // Load this sublayer with its own fresh resolver
                tinyusdz::Layer sub_layer;
                std::string sl_warn, sl_err;
                if (!tinyusdz::LoadLayerFromFile(sl_abs, &sub_layer,
                                                  &sl_warn, &sl_err)) {
                    Log(Warn, "USDScene: failed to load sublayer %s: %s",
                        sl_path.c_str(), sl_err.c_str());
                    continue;
                }

                std::string sl_dir = fs::path(sl_abs).parent_path().string();

                // If this sublayer has its own sublayers, compose them
                if (!sub_layer.metas().subLayers.empty()) {
                    tinyusdz::AssetResolutionResolver sl_resolver;
                    sl_resolver.set_current_working_path(sl_dir);
                    sl_resolver.set_search_paths({sl_dir, base_dir});
                    tinyusdz::Layer sl_comp;
                    if (!tinyusdz::CompositeSublayers(sl_resolver, sub_layer,
                                                       &sl_comp, &sl_warn, &sl_err)) {
                        Log(Warn, "USDScene: nested sublayer composition failed for %s: %s",
                            sl_path.c_str(), sl_err.c_str());
                    } else {
                        sub_layer = std::move(sl_comp);
                    }
                }

                // Create a temporary layer with just this sublayer to use
                // CompositeSublayers for proper over/def merging.
                // We wrap the sub_layer content as a single-sublayer layer.
                tinyusdz::Layer wrapper;
                wrapper.metas() = src_layer.metas();
                // Copy existing primspecs to wrapper
                for (const auto &ps : src_layer.primspecs())
                    wrapper.add_primspec(ps.first, ps.second);
                // Merge sublayer primspecs using OverridePrimSpec for existing
                for (const auto &ps : sub_layer.primspecs()) {
                    if (wrapper.has_primspec(ps.first)) {
                        // Merge: overlay sublayer content onto existing prim
                        tinyusdz::OverridePrimSpec(
                            wrapper.primspecs().at(ps.first), ps.second,
                            &sl_warn, &sl_err);
                    } else {
                        wrapper.add_primspec(ps.first, ps.second);
                    }
                }
                src_layer = std::move(wrapper);

                Log(Info, "USDScene: after sublayer[%zu]: %zu top-level primspecs",
                    i, src_layer.primspecs().size());
            }
            // Clear sublayers list so they aren't processed again
            src_layer.metas().subLayers.clear();
        }

        // 2. Iteratively resolve remaining arcs until stable
        constexpr int kMaxIter = 32;
        for (int iter = 0; iter < kMaxIter; ++iter) {
            bool has_unresolved = false;

            // Inherits (non-fatal: class prims like __class_mtl__ may not exist
            // when material content comes from MaterialX references instead)
            if (src_layer.check_unresolved_inherits()) {
                tinyusdz::Layer comp;
                if (!tinyusdz::CompositeInherits(src_layer,
                                                  &comp, &warn, &err)) {
                    if (iter == 0)
                        Log(Warn, "USDScene: inherits composition skipped (class prims not found)");
                    err.clear();
                } else {
                    has_unresolved = true;
                    src_layer = std::move(comp);
                }
            }

            // References (non-fatal: tinyusdz can't resolve MaterialX
            // references with sub-prim paths like /MaterialX/Materials/name)
            if (src_layer.check_unresolved_references()) {
                tinyusdz::Layer comp;
                if (!tinyusdz::CompositeReferences(resolver, src_layer,
                                                    &comp, &warn, &err)) {
                    if (iter == 0)
                        Log(Warn, "USDScene: reference composition skipped (MaterialX refs unsupported)");
                    err.clear();
                } else {
                    has_unresolved = true;
                    src_layer = std::move(comp);
                }
            }

            // Payload
            if (src_layer.check_unresolved_payload()) {
                has_unresolved = true;
                tinyusdz::Layer comp;
                if (!tinyusdz::CompositePayload(resolver, src_layer,
                                                 &comp, &warn, &err))
                    Throw("USDScene: payload composition failed: %s",
                          err.c_str());
                src_layer = std::move(comp);
            }

            // Variants
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

        // --- Phase 1: Traverse prim tree, collect mesh info ---
        std::vector<MeshInfo> mesh_infos;

        {
            auto identity = tinyusdz::value::matrix4d::identity();
            for (const auto &root_prim : stage.root_prims())
                collect_meshes(root_prim, identity, stage, mesh_infos);
        }

        Log(Info, "USDScene: found %zu meshes, freeing stage...",
            mesh_infos.size());

        // Free the stage and composed layer before creating usdmesh
        // shapes (which will load the file again via the usdmesh cache).
        // This avoids having two copies of the full geometry in memory.
        { tinyusdz::Stage tmp; std::swap(stage, tmp); }
        { tinyusdz::Layer tmp; std::swap(src_layer, tmp); }

        // --- Phase 2: Create shapes from collected mesh info ---
        size_t mesh_count = 0;
        std::string abs_filename = fs::absolute(file_path).string();
        for (auto &mi : mesh_infos) {
            try {
                const MtlxOpenPBRParams *mtlx = lookup_mtlx(mi.mat_name);
                auto bsdf = create_openpbr_bsdf(mtlx, mi.color);

                Properties mesh_props("usdmesh");
                mesh_props.set("filename", abs_filename);
                mesh_props.set("prim_path", mi.prim_path);
                mesh_props.set("to_world", to_mi_transform(mi.world_matrix));
                mesh_props.set("bsdf", ref<Object>(bsdf.get()));

                auto shape = PluginManager::instance()
                    ->create_object<Shape>(mesh_props);
                m_shapes.push_back(ref<Object>(shape.get()));
                mesh_count++;
            } catch (const std::exception &e) {
                Log(Warn, "USDScene: failed to create mesh \"%s\": %s",
                    mi.prim_path.c_str(), e.what());
            }
        }

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

    /// Lazy-lookup MaterialX OpenPBR params for a material name.
    /// Searches for {base_dir}/materials/{name}.mtlx and {base_dir}/{name}.mtlx
    const MtlxOpenPBRParams *lookup_mtlx(const std::string &mat_name) {
        if (mat_name.empty()) return nullptr;

        auto it = m_mtlx_cache.find(mat_name);
        if (it != m_mtlx_cache.end())
            return it->second.valid ? &it->second : nullptr;

        // Try to load from disk
        for (const auto &dir : {m_base_dir + "/materials", m_base_dir}) {
            std::string path = dir + "/" + mat_name + ".mtlx";
            if (fs::exists(path)) {
                MtlxOpenPBRParams params = parse_mtlx_openpbr(path);
                auto result = m_mtlx_cache.insert({mat_name, params});
                if (params.valid) return &result.first->second;
                return nullptr;
            }
        }

        // Not found — cache a negative result
        MtlxOpenPBRParams empty;
        m_mtlx_cache.insert({mat_name, empty});
        return nullptr;
    }

    /// Get the material name from a material:binding path
    /// e.g. "/World/Looks/dresser" → "dresser"
    static std::string material_name_from_path(const std::string &path) {
        auto pos = path.rfind('/');
        if (pos != std::string::npos)
            return path.substr(pos + 1);
        return path;
    }

    /// Create an OpenPBR BSDF from MaterialX parameters or fallback color
    ref<Object> create_openpbr_bsdf(const MtlxOpenPBRParams *mtlx,
                                     ScalarColor3f fallback_color) {
        Properties bsdf_props("openpbr");

        if (mtlx && mtlx->valid) {
            // Use MaterialX parameters
            bsdf_props.set("base_weight", mtlx->base_weight);
            if (mtlx->has_base_color)
                bsdf_props.set("base_color",
                    ScalarColor3f(mtlx->base_color[0], mtlx->base_color[1], mtlx->base_color[2]));
            else
                bsdf_props.set("base_color", fallback_color);
            if (mtlx->has_metalness)
                bsdf_props.set("base_metalness", mtlx->base_metalness);
            if (mtlx->has_specular_weight)
                bsdf_props.set("specular_weight", mtlx->specular_weight);
            if (mtlx->has_specular_color)
                bsdf_props.set("specular_color",
                    ScalarColor3f(mtlx->specular_color[0], mtlx->specular_color[1], mtlx->specular_color[2]));
            if (mtlx->has_specular_roughness)
                bsdf_props.set("specular_roughness", mtlx->specular_roughness);
            if (mtlx->has_specular_ior)
                bsdf_props.set("specular_ior", mtlx->specular_ior);
            if (mtlx->has_transmission)
                bsdf_props.set("transmission_weight", mtlx->transmission_weight);
            if (mtlx->has_coat)
                bsdf_props.set("coat_weight", mtlx->coat_weight);
            if (mtlx->has_coat_color)
                bsdf_props.set("coat_color",
                    ScalarColor3f(mtlx->coat_color[0], mtlx->coat_color[1], mtlx->coat_color[2]));
            if (mtlx->has_coat_roughness)
                bsdf_props.set("coat_roughness", mtlx->coat_roughness);
            if (mtlx->has_coat_ior)
                bsdf_props.set("coat_ior", mtlx->coat_ior);
        } else {
            bsdf_props.set("base_color", fallback_color);
        }

        auto inner = PluginManager::instance()->create_object<BSDF>(bsdf_props);

        // Wrap in twosided so both face sides are visible
        Properties ts_props("twosided");
        ts_props.set("bsdf", ref<Object>(inner.get()));
        return PluginManager::instance()->create_object<BSDF>(ts_props);
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

    /// Get material name from material:binding on a mesh prim
    static std::string get_bound_material_name(
        const tinyusdz::GeomMesh *mesh,
        const tinyusdz::Prim &prim)
    {
        if (mesh->has_materialBinding()) {
            auto &rel = mesh->materialBinding.value();
            auto target = rel.targetPath;
            if (target.is_valid())
                return material_name_from_path(target.prim_part());
        }
        return {};
    }

    /// Walk parent prims to find inherited material:binding
    static std::string find_inherited_material_name(
        const tinyusdz::Prim &prim,
        const tinyusdz::Stage &stage)
    {
        // Check this prim's children for materialBinding
        // (material_assignment.usda puts bindings on intermediate Scope prims)
        if (auto *mesh = prim.as<tinyusdz::GeomMesh>()) {
            if (mesh->has_materialBinding()) {
                auto &rel = mesh->materialBinding.value();
                auto target = rel.targetPath;
                if (target.is_valid())
                    return material_name_from_path(target.prim_part());
            }
        }
        return {};
    }

    /// Collect mesh info (prim path, transform, material) without creating shapes
    struct MeshInfo {
        std::string prim_path;
        tinyusdz::value::matrix4d world_matrix;
        std::string mat_name;
        ScalarColor3f color;
    };

    void collect_meshes(const tinyusdz::Prim &prim,
                        const tinyusdz::value::matrix4d &parent_world,
                        const tinyusdz::Stage &stage,
                        std::vector<MeshInfo> &out) {
        auto local = get_local_xform(prim);
        auto world = local * parent_world;

        if (auto *mesh = prim.as<tinyusdz::GeomMesh>()) {
            auto points = mesh->get_points();
            if (!points.empty()) {
                MeshInfo mi;
                mi.prim_path = prim.absolute_path().full_path_name();
                mi.world_matrix = world;
                mi.mat_name = get_bound_material_name(mesh, prim);
                mi.color = get_material_color(mesh, prim, stage);
                out.push_back(std::move(mi));
            }
        }

        for (const auto &child : prim.children())
            collect_meshes(child, world, stage, out);
    }

    std::vector<ref<Object>> m_shapes;
    std::string m_base_dir;
    std::unordered_map<std::string, MtlxOpenPBRParams> m_mtlx_cache;
};

MI_EXPORT_PLUGIN(USDScene)
NAMESPACE_END(mitsuba)
