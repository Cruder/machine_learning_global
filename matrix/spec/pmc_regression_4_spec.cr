require "./spec_helper"

describe Neuratron::DeepModel do
  it "with 4 exemples" do
    model = Neuratron::DeepModel.new(2, 1, 1)
    result = model.predict([1.0, 2.0])
    pp "result", result
    pp "model", model
  end
end
