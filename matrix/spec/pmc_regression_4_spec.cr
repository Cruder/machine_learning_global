require "./spec_helper"

describe Neuratron::DeepModel do
  it "with 4 exemples" do
    model = Neuratron::DeepModel.new(2, 3, 1)
    pp model
  end
end
