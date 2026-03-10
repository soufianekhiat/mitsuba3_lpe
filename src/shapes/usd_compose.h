// SPDX-License-Identifier: MIT
// Shared USD stage composition helper for usdmesh + usdscene plugins.
// Header-only: each DLL gets its own inline stage cache.
#pragma once

#include "tinyusdz.hh"
#include "composition.hh"
#include "io-util.hh"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

namespace usd_compose {

namespace fs = std::filesystem;

/// Thread-safe cache of composed USD stages, keyed by absolute file path.
inline std::mutex &cache_mutex() {
    static std::mutex m;
    return m;
}
inline std::unordered_map<std::string, std::shared_ptr<tinyusdz::Stage>> &cache_map() {
    static std::unordered_map<std::string, std::shared_ptr<tinyusdz::Stage>> c;
    return c;
}

/// Clear the stage cache (e.g. after usdscene is done discovering meshes).
inline void clear_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex());
    cache_map().clear();
}

/// Compose sublayers of a layer in-place (reverse order, fresh resolvers).
/// Non-fatal: logs warnings via warn_fn but doesn't throw.
inline void compose_sublayers(
    tinyusdz::Layer &src_layer,
    const std::string &base_dir,
    const std::function<void(const std::string &)> &warn_fn = nullptr)
{
    if (src_layer.metas().subLayers.empty())
        return;

    auto warn = [&](const std::string &msg) {
        if (warn_fn) warn_fn(msg);
    };

    size_t n_sl = src_layer.metas().subLayers.size();
    for (size_t ri = 0; ri < n_sl; ++ri) {
        size_t idx = n_sl - 1 - ri;
        const auto &sl = src_layer.metas().subLayers[idx];
        std::string sl_path = sl.assetPath.GetAssetPath();

        // Resolve relative to base_dir
        std::string sl_abs = base_dir + "/" + sl_path;
        {
            auto pos = sl_abs.find("/./");
            while (pos != std::string::npos) {
                sl_abs.erase(pos, 2);
                pos = sl_abs.find("/./");
            }
        }

        if (!fs::exists(sl_abs)) {
            warn("sublayer not found: " + sl_path);
            continue;
        }

        // Load with fresh resolver
        tinyusdz::Layer sub_layer;
        std::string sl_warn, sl_err;
        if (!tinyusdz::LoadLayerFromFile(sl_abs, &sub_layer, &sl_warn, &sl_err)) {
            warn("failed to load sublayer " + sl_path + ": " + sl_err);
            continue;
        }

        std::string sl_dir = tinyusdz::io::GetBaseDir(sl_abs);

        // Recursively compose nested sublayers
        if (!sub_layer.metas().subLayers.empty()) {
            tinyusdz::AssetResolutionResolver sl_resolver;
            sl_resolver.set_current_working_path(sl_dir);
            sl_resolver.set_search_paths({sl_dir, base_dir});
            tinyusdz::Layer sl_comp;
            if (tinyusdz::CompositeSublayers(sl_resolver, sub_layer,
                                              &sl_comp, &sl_warn, &sl_err))
                sub_layer = std::move(sl_comp);
        }

        // Merge sublayer primspecs into src_layer
        for (const auto &ps : sub_layer.primspecs()) {
            if (src_layer.has_primspec(ps.first))
                tinyusdz::OverridePrimSpec(
                    src_layer.primspecs().at(ps.first), ps.second,
                    &sl_warn, &sl_err);
            else
                src_layer.add_primspec(ps.first, ps.second);
        }
    }
    src_layer.metas().subLayers.clear();
}

/// Iteratively resolve LIVRPS arcs (inherits, references, payload, variant).
/// Non-fatal for inherits/references (MaterialX scenes).
inline bool resolve_arcs(
    tinyusdz::Layer &src_layer,
    const std::string &base_dir,
    std::string *err_out,
    const std::function<void(const std::string &)> &warn_fn = nullptr)
{
    tinyusdz::AssetResolutionResolver resolver;
    resolver.set_current_working_path(base_dir);
    std::vector<std::string> search_paths = {base_dir};
    for (const auto &sub : {"cameras", "materials", "assets", "lights",
                             "textures", "geometry"}) {
        std::string sub_dir = base_dir + "/" + sub;
        if (fs::exists(sub_dir))
            search_paths.push_back(sub_dir);
    }
    resolver.set_search_paths(search_paths);

    std::string warn, err;
    constexpr int kMaxIter = 32;
    for (int iter = 0; iter < kMaxIter; ++iter) {
        bool has_unresolved = false;

        // Inherits (non-fatal for MaterialX-based scenes)
        if (src_layer.check_unresolved_inherits()) {
            tinyusdz::Layer comp;
            if (tinyusdz::CompositeInherits(src_layer, &comp, &warn, &err)) {
                has_unresolved = true;
                src_layer = std::move(comp);
            }
            err.clear();
        }

        // References (non-fatal for MaterialX references)
        if (src_layer.check_unresolved_references()) {
            tinyusdz::Layer comp;
            if (tinyusdz::CompositeReferences(resolver, src_layer,
                                                &comp, &warn, &err)) {
                has_unresolved = true;
                src_layer = std::move(comp);
            }
            err.clear();
        }

        // Payload
        if (src_layer.check_unresolved_payload()) {
            has_unresolved = true;
            tinyusdz::Layer comp;
            if (!tinyusdz::CompositePayload(resolver, src_layer,
                                             &comp, &warn, &err)) {
                if (err_out) *err_out = "payload composition failed: " + err;
                return false;
            }
            src_layer = std::move(comp);
        }

        // Variants
        if (src_layer.check_unresolved_variant()) {
            has_unresolved = true;
            tinyusdz::Layer comp;
            if (!tinyusdz::CompositeVariant(src_layer, &comp, &warn, &err)) {
                if (err_out) *err_out = "variant composition failed: " + err;
                return false;
            }
            src_layer = std::move(comp);
        }

        if (!has_unresolved) break;
    }
    return true;
}

/// Load, compose, and cache a USD stage.  Thread-safe.
inline std::shared_ptr<tinyusdz::Stage> load_composed_stage(
    const std::string &abs_path,
    const std::string &base_dir,
    std::string *err_out,
    const std::function<void(const std::string &)> &warn_fn = nullptr)
{
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &cmap = cache_map();
    auto it = cmap.find(abs_path);
    if (it != cmap.end())
        return it->second;

    tinyusdz::Layer root_layer;
    std::string warn, err;
    if (!tinyusdz::LoadLayerFromFile(abs_path, &root_layer, &warn, &err)) {
        if (err_out) *err_out = "load failed: " + err;
        return nullptr;
    }

    tinyusdz::Layer src_layer = root_layer;
    compose_sublayers(src_layer, base_dir, warn_fn);
    if (!resolve_arcs(src_layer, base_dir, err_out, warn_fn))
        return nullptr;

    auto stage = std::make_shared<tinyusdz::Stage>();
    if (!tinyusdz::LayerToStage(src_layer, stage.get(), &warn, &err)) {
        if (err_out) *err_out = "LayerToStage failed: " + err;
        return nullptr;
    }
    stage->compute_absolute_prim_path_and_assign_prim_id();

    cmap[abs_path] = stage;
    return stage;
}

} // namespace usd_compose
