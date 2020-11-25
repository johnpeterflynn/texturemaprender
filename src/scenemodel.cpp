#include "scenemodel.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

SceneModel::SceneModel(string const &model_path, string const &aggregation_path,
                       string const &segs_path, int mask)
    : Model(model_path, mask)
    , m_aggregation_path(aggregation_path)
    , m_segs_path(segs_path)
{
    //"resources/scan0/scene0000_00_vh_clean.aggregation.json"
    //"resources/scan0/scene0000_00_vh_clean_smartuv_75_0_0_no_aspect.segs.json"

}

std::vector<std::string> SceneModel::loadSegmentLabels() {
    std::ifstream agg_s(m_aggregation_path);
    json jagg = json::parse(agg_s);
    auto seg_groups = jagg["segGroups"];

    // TODO: NOTE: We are implicitly assuming that each list entry's indes is also its id
    std::vector<std::string> labels;
    for (int i = 0; i < seg_groups.size(); i++) {
        auto label = seg_groups[i]["label"];
        labels.push_back(label);
    }

    return labels;
}

std::shared_ptr<Model> SceneModel::extractLabeledSubmodel(int id) {
    std::ifstream agg_s(m_aggregation_path);
    json jagg = json::parse(agg_s);
    // 35 - table near chair. move to (-0.6, -1, 0)
    // 12 - trash can by fridge. move to (0.13, -0.4, 0)
    std::vector<int> segs = jagg["segGroups"][id]["segments"];
    std::ifstream seg_s(m_segs_path);
    json jseg = json::parse(seg_s);
    std::vector<int> seg_indices = jseg["segIndices"];

    std::cout << "Getting segmentation vertex indices of label " << id << " from segmentation groups\n";

    // Get segmentation vertex indices with respect to the scene mesh.
    std::vector<int> seg_vert_indices;
    std::unordered_set<int> seg_vert_indices_hash;
    for (size_t i = 0; i < seg_indices.size(); i++) {
        if (std::find(segs.begin(), segs.end(), seg_indices[i]) != segs.end()) {
            seg_vert_indices.push_back(i);
            seg_vert_indices_hash.insert(i);
        }
    }

    vector<Vertex> vertices;
    vector<unsigned int> indices;
    vector<Texture> textures;
    auto &mesh = meshes[0];

    std::cout << "Copying segmentation faces and vertices from scene model\n";

    // Compute segmentation vertex indices for each submesh face with respect to
    //  the submesh.
    int vertices_per_face = 3;
    for(unsigned int i = 0; i < mesh.indices.size(); i+=vertices_per_face)
    {
        // save vertex indices for all faces made up of only segmentation
        //  vertices
        bool b_all_seg_verts = true;

        // Flag faces with all vertices in seg_vert_indices
        for(unsigned int j = 0; j < vertices_per_face; j++) {
            int mesh_face_vert_index = mesh.indices[i + j];
            if(seg_vert_indices_hash.find(mesh_face_vert_index) == seg_vert_indices_hash.end()) {
                b_all_seg_verts = false;
            }
        }

        if (b_all_seg_verts) {
            for(unsigned int j = 0; j < vertices_per_face; j++) {
                int mesh_face_vert_index = mesh.indices[i + j];
                auto it = std::find(seg_vert_indices.begin(), seg_vert_indices.end(),
                                    mesh_face_vert_index);
                int index = std::distance(seg_vert_indices.begin(), it);
                indices.push_back(index);
            }
        }
    }

    // Get all vertices from scene mesh within segmentation
    // TODO: Find center of mesh instead of mean vertex
    // TODO: Use a more compact vector representation
    Vertex mean_vertex;
    mean_vertex.Position.x = 0;
    mean_vertex.Position.y = 0;
    mean_vertex.Position.z = 0;
    for(unsigned int i = 0; i < seg_vert_indices.size(); i++) {
        Vertex vertex = mesh.vertices[seg_vert_indices[i]];

        mean_vertex.Position.x += vertex.Position.x;
        mean_vertex.Position.y += vertex.Position.y;
        mean_vertex.Position.z += vertex.Position.z;

        vertices.push_back(vertex);
    }

    mean_vertex.Position.x /= seg_vert_indices.size();
    mean_vertex.Position.y /= seg_vert_indices.size();
    mean_vertex.Position.z /= seg_vert_indices.size();

    // Subtract mean from every vertex
    for(auto &vertex : vertices) {
        vertex.Position.x -= mean_vertex.Position.x;
        vertex.Position.y -= mean_vertex.Position.y;
        vertex.Position.z -= mean_vertex.Position.z;
    }

    // TODO: This is sort of a hack. Handle lifecycle of model automatically.
    std::shared_ptr<Model> submodel = std::make_shared<Model>();

    // Position submodel at its previous mean vector position
    submodel->m_position[0] = mean_vertex.Position.x;
    submodel->m_position[1] = mean_vertex.Position.y;
    submodel->m_position[2] = mean_vertex.Position.z;

    submodel->meshes.emplace_back(vertices, indices, textures);
    return submodel;
}
