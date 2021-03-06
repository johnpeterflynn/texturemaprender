#ifndef SCENEMODEL_H
#define SCENEMODEL_H

#include "model.h"

class SceneModel : public Model {
public:
    SceneModel(string const &model_path, string const &aggregation_path,
               string const &segs_path, int mask = Model::DEFAULT_MASK);

    std::vector<std::string> loadSegmentLabels();
    std::shared_ptr<Model> extractLabeledSubmodel(int id);
    void deleteLabeledSubmodel(int id);

private:
    std::string m_aggregation_path;
    std::string m_segs_path;
};

#endif // SCENEMODEL_H
