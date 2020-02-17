@[Link("neuratron")]
lib LibLibrary
  fun give_me_42 = give_42() : Int32
end

class Giver42
  def initialize
    @value = LibLibrary.give_me_42
  end

  def give
    @value
  end
end

pp LibLibrary.give_me_42

pp Giver42.new.give
