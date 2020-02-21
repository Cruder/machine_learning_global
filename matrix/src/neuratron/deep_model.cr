module Neuratron
  class DeepModel
    @model : LibNeuratron::DeepModel*

    def initialize(layers)
      @model = LibNeuratron.create_deep_model(layers.to_unsafe, layers.size)
    end

    def initialize(*layers : Int)
      initialize(layers.to_a)
    end

    def predict(input)
      result = LibNeuratron.predict_deep_model_regression(@model, input)
      Array.new(@model.value.d[@model.value.layer_count - 1] - 1) { |i| result[i + 1] }
    end

    def train_regression(inputs : Array(Array(Float64)), outputs : Array(Array(Float64)), iteration : Int32, training_rate : Float64)
        zip_original = inputs.zip(outputs)
        (1..iteration).each { |i|
          puts "Iteration #{i}"
            shuffled_inputs, shuffled_outputs = shuffle_dataset(zip_original)
            went_well = LibNeuratron.train_deep_regression_model(@model, shuffled_inputs.to_unsafe, shuffled_inputs.size,
                            shuffled_outputs.to_unsafe, shuffled_outputs.size,
                            training_rate)
            false if !went_well
        }
    end

    def shuffle_dataset(zipped_dataset)
        zip_shuffled = zipped_dataset.shuffle
        shuffled_inputs = zip_shuffled.map do |input_output|
            input_output[0]
        end.flatten
        shuffled_outputs = zip_shuffled.map do |input_output|
            input_output[1]
        end.flatten
#        pp shuffled_inputs
#        pp shuffled_outputs
        {shuffled_inputs, shuffled_outputs}
    end

    def train_classification(inputs, outputs, iteration, training_rate)
      pp inputs.size
      pp outputs.size
      zip_original = inputs.zip(outputs)
        (1..iteration).each { |i|
          puts "Iteration #{i}"
            shuffled_inputs, shuffled_outputs = shuffle_dataset(zip_original)
            went_well = LibNeuratron.train_deep_classification_model(@model,
                            shuffled_inputs.to_unsafe, shuffled_inputs.size,
                            shuffled_outputs.to_unsafe, shuffled_outputs.size,
                            training_rate)
        }
        true
    end

    def generate_expected_indice(indice) : Array(Float64)
      Array.new(@model.value.d[@model.value.layer_count - 1] - 1) do |i|
        i == indice ? 1.0 : -1.0
      end
    end

    def save(filename)
      data = {
        layer_count: @model.value.layer_count,
        d: d,
        w: w,
      }.to_json

      File.write(filename, data)
    end

    private def w
      Array.new(@model.value.layer_count - 1) do |i|
        Array.new(d[i]) do |j|
          Array.new(d[i + 1]) do |k|
            @model.value.w[i][j][k]
          end
        end
      end
    end

    private def d
      Array.new(@model.value.layer_count) { |i| @model.value.d[i] }
    end
  end
end
