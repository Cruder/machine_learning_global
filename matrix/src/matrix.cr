require "aquaplot"

@[Link("neuratron")]
lib LibNeuratron
  @[Extern]
  struct LinearModel
    inputs: Float64*
    size_input: Int32
    size_output: Int32
  end

  @[Extern]
  struct DeepModel
    deltas : Float64**
    x : Float64**
    w : Float64***
    d : Int32*
  end

  fun create_linear_model(input : Int32, size: Int32) : Pointer(LinearModel)
  fun train_linear_model(
    model : LinearModel*,
    input : Float64*, input_size : Int32,
    output : Float64*, output_size : Int32
  ) : Bool

  fun predict_linear_model(model : LinearModel*, input : Float64*) : Float64
  fun free_model(model : Pointer(LinearModel)) : Int32
end

module Neuratron
  class LinearModel
    class Regression
      def predict(model, input)
        LibNeuratron.predict_linear_model(model, input)
      end
    end

    class Classification
      def predict(model, input)
        LibNeuratron.predict_linear_model(model, input)
      end
    end

    @model : LibNeuratron::LinearModel*

    def initialize(@input_size : Int32, @ouput_size : Int32)
      @model = LibNeuratron.create_linear_model(input_size, ouput_size)
    end

    def finalize
      LibNeuratron.free_model(@model)
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
end
