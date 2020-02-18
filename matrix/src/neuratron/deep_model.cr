module Neuratron
  class DeepModel
    class Regression
      def predict(model, input)
        LibNeuratron.predict_deep_model_regression(model, input)
      end
    end

    class Classification
      def predict(model, input)
        LibNeuratron.predict_deep_model_classification(model, input)
      end
    end

    @model : LibNeuratron::DeepModel*

    def initialize(layers)
      @model = LibNeuratron.create_deep_model(layers.to_unsafe, layers.size)
    end

    def initialize(*layers : Int)
      initialize(layers.to_a)
    end

    def predict(input, kind = Regression.new)
      kind.predict(@model, input)
    end
  end
end
