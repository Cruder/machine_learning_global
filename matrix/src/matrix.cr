require "aquaplot"

@[Link("neuratron")]
lib LibNeuratron
  @[Extern]
  struct DeepModel
    deltas : Float64**
    x : Float64**
    w : Float64***
    d : Int32*
  end

  fun create_deep_model(neurons_per_layers : Int32*, size : Int32) : DeepModel*
  fun train_deep_model(
    model : DeepModel*,
    input : Float64*, input_size : Int32,
    output : Float64*, output_size : Int32
  ) : Bool
  fun predict_deep_model_regression(model : DeepModel*, input : Float64*) : Float64
  fun predict_deep_model_classification(model : DeepModel*, input : Float64*) : Float64
  fun free_deep_model(model : DeepModel*) : Int32

  @[Extern]
  struct LinearModel
    inputs: Float64*
    size_input: Int32
    size_output: Int32
  end

  fun create_linear_model(input : Int32, size: Int32) : Pointer(LinearModel)
  fun train_linear_model(
    model : LinearModel*,
    input : Float64*, input_size : Int32,
    output : Float64*, output_size : Int32
  ) : Bool

  fun predict_linear_model_regression(model : LinearModel*, input : Float64*) : Float64
  fun predict_linear_model_classification(model : LinearModel*, input : Float64*) : Float64
  fun free_linear_model(model : Pointer(LinearModel)) : Int32
end

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
      kind.predict(@model, input)
    end
  end

  class DeepModel
    @model : LibNeuratron::DeepModel*

    def initialize(layers)
      @model = LibNeuratron.create_deep_model(layers.to_unsafe, layers.size)
    end

    def initialize(*layers : Int)
      initialize(layers.to_a)
    end
  end
end
