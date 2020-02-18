module Neuratron
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
