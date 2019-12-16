
require "big"


STRETCH_CONSTANT_2D = -0.211324865405187    # (1/Math.sqrt(2+1)-1)/2
SQUISH_CONSTANT_2D = 0.366025403784439      # (Math.sqrt(2+1)-1)/2
NORM_CONSTANT_2D = 47
DEFAULT_SEED = 0_i32
GRADIENTS_2D = [5, 2, 2, 5, -5, 2, -2, 5, 5, -2, 2, -5, -5, -2, -2, -5]

class OpenSimplex
  # """
  # OpenSimplex n-dimensional gradient noise functions.
  # """

  property seed : Int32
  property perm : Array(Int32)
  property source : Array(Int32)

  def overflow(num : Int32) : Int32
    buffer = BigInt.new num
    buffer = buffer * 6364136223846793005 + 1442695040888963407
    return buffer.to_i32!
  end
  
  def initialize(seed : Int32 = DEFAULT_SEED)
    # """
    # Initiate the class and generate permutation arrays from a seed number.
    # """
    # Initializes the class using a permutation array generated from a 64-bit seed.
    # Generates a proper permutation (i.e. doesn't merely perform N
    # successive pair swaps on a base array)
    
    @seed = seed
    @perm = Array(Int32).new 256, 0 # Have to zero fill so we can properly loop over it later
    @source = (0..255).to_a
    
    3.times do
      @seed = overflow @seed
    end

    @source.reverse.each do |i|
      @seed = overflow @seed
      r = (@seed + 31) % (i + 1)
      if r < 0
        r += i + 1
      end
      @perm[i] = @source[r]
      @source[r] = @source[i]
    end
  end

  def set_seed(value : Int32)
      initialize value
  end

  def smooth_noise(x, y, limit=1, scale=10.0, octaves=2, persistence=1.2)
    total = 0
    frequency = 1
    amplitude = 1
    max_value = 0
    octaves.times do |octave|
      total += scale_noise2(x * frequency / scale, y * frequency / scale, limit) * amplitude
      max_value += amplitude
      amplitude *= persistence
      frequency *= 2
    end

    return total / max_value
  end

  def get_noise(x, y, limit=1)
    value = noise2d(x, y) + 0.86435
    return limit / 1.6415 * value
  end

  def extrapolate(xsb, ysb, dx, dy)
    perm = @perm
    index = perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E

    g1, g2 = GRADIENTS_2D[(index..index + 1)]
    return g1 * dx + g2 * dy
  end

  def noise2d(x, y)
    # """
    # Generate 2D OpenSimplex noise from X,Y coordinates.
    # """
    # Place input coordinates onto grid.
    stretch_offset = (x + y) * STRETCH_CONSTANT_2D
    xs = x + stretch_offset
    ys = y + stretch_offset

    # Floor to get grid coordinates of rhombus (stretched square) super-cell origin.
    xsb = xs.floor.to_i
    ysb = ys.floor.to_i

    # Skew out to get actual coordinates of rhombus origin. We'll need these later.
    squish_offset = (xsb + ysb) * SQUISH_CONSTANT_2D
    xb = xsb + squish_offset
    yb = ysb + squish_offset

    # Compute grid coordinates relative to rhombus origin.
    xins = xs - xsb
    yins = ys - ysb

    # Sum those together to get a value that determines which region we're in.
    in_sum = xins + yins

    # Positions relative to origin point.
    dx0 = x - xb
    dy0 = y - yb

    value = 0

    # Contribution (1,0)
    dx1 = dx0 - 1 - SQUISH_CONSTANT_2D
    dy1 = dy0 - 0 - SQUISH_CONSTANT_2D
    attn1 = 2 - dx1 * dx1 - dy1 * dy1
    if attn1 > 0
      attn1 *= attn1
      value += attn1 * attn1 * extrapolate(xsb + 1, ysb + 0, dx1, dy1)
    end

    # Contribution (0,1)
    dx2 = dx0 - 0 - SQUISH_CONSTANT_2D
    dy2 = dy0 - 1 - SQUISH_CONSTANT_2D
    attn2 = 2 - dx2 * dx2 - dy2 * dy2
    if attn2 > 0
      attn2 *= attn2
      value += attn2 * attn2 * extrapolate(xsb + 0, ysb + 1, dx2, dy2)
    end

    if in_sum <= 1 # We're inside the triangle (2-Simplex) at (0,0)
      zins = 1 - in_sum
      if zins > xins || zins > yins # (0,0) is one of the closest two triangular vertices
        if xins > yins
          xsv_ext = xsb + 1
          ysv_ext = ysb - 1
          dx_ext = dx0 - 1
          dy_ext = dy0 + 1
        else
          xsv_ext = xsb - 1
          ysv_ext = ysb + 1
          dx_ext = dx0 + 1
          dy_ext = dy0 - 1
        end
      else # (1,0) and (0,1) are the closest two vertices.
        xsv_ext = xsb + 1
        ysv_ext = ysb + 1
        dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
        dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT_2D
      end
    else # We're inside the triangle (2-Simplex) at (1,1)
      zins = 2 - in_sum
      if zins < xins || zins < yins # (0,0) is one of the closest two triangular vertices
        if xins > yins
          xsv_ext = xsb + 2
          ysv_ext = ysb + 0
          dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT_2D
          dy_ext = dy0 + 0 - 2 * SQUISH_CONSTANT_2D
        else
          xsv_ext = xsb + 0
          ysv_ext = ysb + 2
          dx_ext = dx0 + 0 - 2 * SQUISH_CONSTANT_2D
          dy_ext = dy0 - 2 - 2 * SQUISH_CONSTANT_2D
        end
      else # (1,0) and (0,1) are the closest two vertices.
        dx_ext = dx0
        dy_ext = dy0
        xsv_ext = xsb
        ysv_ext = ysb
      end
      xsb += 1
      ysb += 1
      dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
      dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT_2D
    end

    # Contribution (0,0) or (1,1)
    attn0 = 2 - dx0 * dx0 - dy0 * dy0
    if attn0 > 0
      attn0 *= attn0
      value += attn0 * attn0 * extrapolate(xsb, ysb, dx0, dy0)
    end

    # Extra Vertex
    attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
    if attn_ext > 0
      attn_ext *= attn_ext
      value += attn_ext * attn_ext * extrapolate(xsv_ext, ysv_ext, dx_ext, dy_ext)
    end

    return value / NORM_CONSTANT_2D
  end

end#class
