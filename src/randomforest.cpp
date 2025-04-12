#include <cstdlib>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/core/cv/metrics/precision.hpp>
#include <mlpack/core/cv/metrics/recall.hpp>
#include <mlpack/core/cv/metrics/f1.hpp>

void usage(char pname[]);

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::cv;

int main(int argc, char *argv[]){
  arma::mat data;

  if (argc < 2) {
    usage(argv[0]);
    return(EXIT_SUCCESS);
  }

  bool loaded = data::Load(argv[1], data);
  if (!loaded) {
    std::cout << "Erro ao abrir \'" << argv[1];
    std::cout << "\'. Arquivo não encontrado ou com formato inválido.\n";
    return EXIT_FAILURE;
  }

  arma::Row<size_t> labels;
  labels = arma::conv_to<arma::Row<size_t>>::from(data.row(data.n_rows -1));
  data.shed_row(data.n_rows -1);

  size_t nclasses = 2;
  size_t minleafsize = 5;
  size_t ntrees = 10;
  double tsplit = 0.25;
  double accuracy, precision, recall, f1;
  RandomForest<GiniGain, RandomDimensionSelect> classifier;


  if (argc > 2) nclasses = (size_t) atoi(argv[2]);
  if (argc > 3) minleafsize = (size_t) atoi(argv[3]);
  if (argc > 4) ntrees = (size_t) atoi(argv[4]);

  // classifier = RandomForest<GiniGain, RandomDimensionSelect>(
  //     data, labels, nclasses, ntrees, minleafsize
  //     );
  SimpleCV<RandomForest<GiniGain, RandomDimensionSelect>, Accuracy>
          crossvalidation(tsplit, data, labels, nclasses);
  accuracy = crossvalidation.Evaluate(ntrees, minleafsize); // training

  std::cout << "RandomForest"
            << "\n  número de classes: " << nclasses
            << "\n  tamanho mínino das folhas: " << minleafsize
            << "\n  número de árvores: " << ntrees
            << "\n";

  classifier = crossvalidation.Model();

  precision = Precision<Binary>::Evaluate(classifier, data, labels);
  recall = Recall<Binary>::Evaluate(classifier, data, labels);
  f1 = F1<Binary>::Evaluate(classifier, data, labels);

  std::cout << "Training Accuracy: " << accuracy
            << "\nPrecision: " << precision
            << "\nRecall: " << recall
            << "\nF1: " << f1 << "\n";

  data::Save("randomforest_model.xml", "model", classifier, false);



  // ----------------------------------
  // para classificar uma nova amostra:
  //
  // arma::Row<size_t> predictions; arma::mat probabilities;
  // data::Load("mymodel.xml", "model", classifier);
  // classifier.Classify(sample, predictions, probabilities); // classification
  // const size_t correct = arma::accu(predictions == labels);

  return 0;
}

void usage(char pname[]){
  std::cout << "Usage: " << pname << " PATH [NCLASSES [MINLEAFSIZE [NTREES]]]\n";
  std::cout << "  Utiliza o método random_forest no dataset em PATH.\n";
}
