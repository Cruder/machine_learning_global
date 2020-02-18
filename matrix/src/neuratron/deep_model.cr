module Neuratron
  class DeepModel
    @model : LibNeuratron::DeepModel*

    def initialize(layers)
      @model = LibNeuratron.create_deep_model(layers.to_unsafe, layers.size)
    end

    def initialize(*layers : Int)
      initialize(layers.to_a)
    end

    def train_regression(input, output)
      LibNeuratron.train_deep_regression_model(@model, input.to_unsafe, input.size, output.to_unsafe, output.size)
    end
  end
end
