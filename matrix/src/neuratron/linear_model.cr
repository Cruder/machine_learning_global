module Neuratron
  class LinearModel
    class Regression
      def predict(model, input)
        LibNeuratron.predict_linear_model_regression(model, input)
      end

      def train(model, input, output)
        LibNeuratron.train_linear_model(model, input.to_unsafe, input.size, output.to_unsafe, output.size)
      end
    end

    class Classification
      def initialize(@alpha = 0.0, @iteration = 1)
      end

      def predict(model, input)
        LibNeuratron.predict_linear_model_classification(model, input)
      end

      def train(model, inputs, outputs)
        zip_original = inputs.zip(outputs)
        pp zip_original[0][1]


        (0..@iteration).each do |i|
          puts "Iteration #{i}"
          zip_original.shuffle.each do |input, output|
            LibNeuratron.train_linear_model_classification(model, input.to_unsafe, output.to_unsafe, @alpha)
          end
        end
      end
    end

    @model : LibNeuratron::LinearModel*

    def initialize(input_size : Int32, ouput_size : Int32)
      @model = LibNeuratron.create_linear_model(input_size, ouput_size)
    end

    def initialize(@model : LibNeuratron::LinearModel*)
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

    def train(input, output, kind = Regression.new)
      kind.train(@model, input, output)
    end

    def predict(input, kind = Regression.new)
      result = kind.predict(@model, input.to_unsafe)

      Array.new(@model.value.size_output) { |i| result[i] }
    end

    def save(filename)
      data = {
        input: @model.value.size_input,
        output: @model.value.size_output,
        weights: weights
      }.to_h.to_json

      File.write(filename, data)
    end
  end
end
