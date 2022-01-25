#include <cstdlib>
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_svm/linear_svm.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/core/cv/metrics/precision.hpp>
#include <mlpack/core/cv/metrics/recall.hpp>
#include <mlpack/core/cv/metrics/f1.hpp>

void usage(char pname[]);

using namespace mlpack;
using namespace mlpack::svm;
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
  double tsplit = 0.25;
  double lambda = 0.001; // L2 regularization
  double delta = 1.0; // margin of difference
  double accuracy, precision, recall, f1;
  LinearSVM<> classifier;


  if (argc > 2) nclasses = (size_t) atoi(argv[2]);
  if (argc > 3) lambda = (size_t) atoi(argv[3]);
  if (argc > 4) delta = (size_t) atoi(argv[4]);

  // classifier = RandomForest<GiniGain, RandomDimensionSelect>(
  //     data, labels, nclasses, ntrees, minleafsize
  //     );
  SimpleCV<LinearSVM<>, Accuracy>
          crossvalidation(tsplit, data, labels, nclasses);
  accuracy = crossvalidation.Evaluate(lambda, delta); // training

  std::cout << "LinearSVM"
            << "\n  número de classes: " << nclasses
            << "\n  constante de regularização L2: " << lambda
            << "\n  margem: " << delta
            << "\n";

  classifier = crossvalidation.Model();

  precision = Precision<Binary>::Evaluate(classifier, data, labels);
  recall = Recall<Binary>::Evaluate(classifier, data, labels);
  f1 = F1<Binary>::Evaluate(classifier, data, labels);

  std::cout << "Training Accuracy: " << accuracy
            << "\nPrecision: " << precision
            << "\nRecall: " << recall
            << "\nF1: " << f1 << "\n";

  data::Save("model.xml", "model", classifier, false);



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
  std::cout << "Usage: " << pname << " PATH [NCLASSES [LAMBDA [DELTA]]]\n";
  std::cout << "  Utiliza o método random_forest no dataset em PATH.\n";
}
