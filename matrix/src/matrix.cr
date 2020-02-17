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
    @model : LinearModel*

    def initialize(@input : Int32, @ouput : Int32)
      @model = LibNeuratron.create_linear_model(input, ouput)
    end

    def model
      @model.value
    end
  end
end

inputs = [
  [0.0, 0.0],
  [1.0, 0.0],
  [1.0, 1.0],
  [0.0, 1.0]
]


expected_outputs = [0, 0, 1, 1]
model = TronLinearModel.new(inputs)
pp model.train
