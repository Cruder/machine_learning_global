require "aquaplot"
require "./neuratron/lib"
require "./neuratron/**"


def max_indice(array : Array(Float64))
  max = array[0];
  index = 0;
  array.each_with_index do |e, i|
    if e > max
      max = e
      index = i
    end
  end
  index
end
