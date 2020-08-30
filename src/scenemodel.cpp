#include "scenemodel.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

SceneModel::SceneModel(string const &model_path, string const &aggregation_path,
                       string const &segs_path)
    : Model(model_path)
    , m_aggregation_path(aggregation_path)
    , m_segs_path(segs_path)
{

}

Model SceneModel::extractLabeledSubmodel(int id) {

}