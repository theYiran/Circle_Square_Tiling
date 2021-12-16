# reference ==> 

import taichi as ti

ti.init(arch=ti.cuda)

res_x = 800
res_y = 512
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))


@ti.func
def fract(vec):
    return vec - ti.floor(vec)


@ti.func
def step(edge, v):
    ret = 0.0
    if (v < edge):
        ret = 0.0
    else:
        ret = 1.0
    return ret


@ti.func
def smoothstep(edge1, edge2, v):
    assert (edge1 != edge2)
    t = (v - edge1) / float(edge2 - edge1)
    t = clamp(t, 0.0, 1.0)

    return (3 - 2 * t) * t ** 2


@ti.func
# 0到1取原值，大于1取1
def clamp(v, v_min, v_max):
    return ti.min(ti.max(v, v_min), v_max)


@ti.func
def circle(pos, center, radius, blur):
    r = (pos - center).norm()
    t = 0.0
    if blur > 1.0: blur = 1.0
    if blur <= 0.0:
        t = 1.0 - step(1.0, r / radius)
    else:
        t = smoothstep(1.0, 1.0 - blur, r / radius)
    return t


@ti.func
def square(pos, center, radius, blur):
    diff = ti.abs(pos - center)
    r = ti.max(diff[0], diff[1])
    t = 0.0
    if blur > 1.0: blur = 1.0
    if blur <= 0.0:
        t = 1.0 - step(1.0, r / radius)
    else:
        t = smoothstep(1.0, 1.0 - blur, r / radius)
    return t


@ti.kernel
def render(t: ti.f32):
    # draw something on your canvas
    for i_, j_ in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])  # init your canvas to black

        tile_size = 3

        i = i_ - t * 10
        j = j_ + t * 10

        for k in range(6):

            center = ti.Vector([tile_size / 2, tile_size / 2])
            radius = tile_size / 2
            pos = ti.Vector([i % tile_size, j % tile_size])  # scale i, j to [0, tile_size-1]

            c=0.0
            blur = fract(ti.sin(float(0.1 * t + i // tile_size * 5 + j // tile_size * 3))) + 0.1
            if (k % 2 == 0):
                c = square(pos, center, radius, blur)
            if (k % 2 == 1):
                c = circle(pos, center, radius, blur)

            r = 0.3 * ti.sin(float(0.001 * t + i // tile_size) + 6) + 0.7
            g = 0.1 * ti.cos(float(0.001 * t + j // tile_size) + 2) + 0.9
            b = 0.3 * ti.sin(float(0.001 * t + i // tile_size) + 4) + 0.7

            color += ti.Vector([r, g, b]) * c

            color /= 2
            tile_size *= 2

        pixels[i_, j_] = color


gui = ti.GUI("Canvas", res=(res_x, res_y))

for i in range(100000):
    t = i * 0.03
    render(t)
    gui.set_image(pixels)
    gui.show()
