@[Link("neuratron")]
lib LibNeuratron
  @[Extern]
  struct LinearModel
    inputs: Float64*
    size_input: Int32
    size_output: Int32
  end

  fun create_linear_model(input : Int32, size: Int32) : Pointer(LinearModel)
  fun train_linear_model = train_linear_model(i : Int32) : Int32
end


module Neuratron
  class LinearModel
    @model : LibNeuratron::LinearModel*

    def initialize(@input_size : Int32, @ouput_size : Int32)
      @model = LibNeuratron.create_linear_model(input_size, ouput_size)
    end

    def model
      @model.value
    end

    def train
      LibNeuratron.train_linear_model(@model)
    end
  end
end

inputs = [
  [0.0, 0.0],
  [1.0, 0.0],
  [1.0, 1.0],
  [0.0, 1.0]
]
expected_outputs = [[0], [0], [1], [1]]
w = [-0.2, 0.5, 0.8]

model = Neuratron::LinearModel.new(2, 1)
pp model.train(input, ouput, w)
