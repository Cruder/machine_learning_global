module Neuratron
  class LinearModel
    class Regression
      def predict(model, input)
        LibNeuratron.predict_linear_model_regression(model, input)
      end
    end

    class Classification
      def predict(model, input)
        LibNeuratron.predict_linear_model_classification(model, input)
      end
    end

    @model : LibNeuratron::LinearModel*

    def initialize(@input_size : Int32, @ouput_size : Int32)
      @model = LibNeuratron.create_linear_model(input_size, ouput_size)
    end

    def finalize
      LibNeuratron.free_linear_model(@model)
    end

    def model
      @model.value
    end

    def weights
      (0...((model.size_input + 1) * model.size_output)).map do |i|
        model.inputs[i]
      end
    end

    def train(input, output)
      LibNeuratron.train_linear_model(@model, input.to_unsafe, input.size, output.to_unsafe, output.size)
    end

    def predict(input, kind = Regression.new)
      result = kind.predict(@model, input.to_unsafe)

      Array.new(@model.value.size_output) { |i| result[i] }
    end
  end
end
