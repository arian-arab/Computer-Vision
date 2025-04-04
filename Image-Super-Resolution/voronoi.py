from PIL import Image, ImageColor, ImageDraw
from random import randint, choice, random, seed as randomseed
from typing import *
from math import hypot, sqrt
from queue import Queue
from dataclasses import dataclass
from enum import Enum
import numpy as np
import os
import sys


class ColorAlgorithm(Enum):
    random           = 1
    no_adjacent_same = 2
    least_possible   = 3


class RegionAlgorithm:
    def randomized(width: int, height: int, regions: int, mask_function) -> List[Tuple[int, int]]:
        """Return regions that are entirely random."""
        points = []
        while len(points) != regions:
            p = (randint(0, width - 1), randint(0, height - 1))

            if p in points:
                continue

            if not mask_function(p):
                continue

            points.append(p)

        return points

    def uniform(width: int, height: int, regions: int, mask_function) -> List[Tuple[int, int]]:
        """Return regions that attempt to be somewhat uniform."""
        k = 10
        points = []
        while len(points) != regions:
            best_p = None
            d_max = 0

            i = 0
            while i < k * len(points) + 1:
                p = (randint(0, width - 1), randint(0, height - 1))

                if p in points:
                    continue

                if not mask_function(p):
                    continue

                if len(points) == 0:
                    best_p = p
                    break

                d_min = float('inf')
                for x, y in points:
                    d = hypot(p[0]-x, p[1]-y)

                    if d < d_min:
                        d_min = d

                if d_min > d_max:
                    d_max = d_min
                    best_p = p

                i += 1

            points.append(best_p)

        return points


class DistanceAlgorithm:
    def euclidean(x, y, xn, yn):
        """Calculate the image regions (up to a distance) using euclidean distance."""
        return hypot(xn-x, yn-y)

    def manhattan(x, y, xn, yn):
        """Calculate the image regions using manhattan distance."""
        return abs(xn-x) + abs(yn-y)

    def euclidean45degrees(x, y, xn, yn):
        """Calculate the image regions using euclidean, but allow only lines in 45 degree increments."""
        return sqrt(2 * min(abs(xn-x), abs(yn-y)) ** 2) + abs(abs(xn-x) - abs(yn-y))

    def chebyshev(x, y, xn, yn):
        """Calculate the image regions using chebyshev distance."""
        return min(abs(xn-x), abs(yn-y)) + abs(abs(xn-x) - abs(yn-y))

    def set_each_point(seed: int, width: int, height: int,
            region_centers: List[Tuple[int, int]], image: List[List[int]],
            d_limit: int, f: List[Callable[[int, int, int, int], float]], mask_function):
        """Calculate the image regions (up to a distance) using the provided metric."""
        randomseed(seed)

        region_distance_functions = [f if not isinstance(f, list) else choice(f) for _ in range(len(region_centers))]

        for x in range(width):
            for y in range(height):
                if not mask_function((x, y)):
                    continue

                d_min = float('inf')

                for i, region in enumerate(region_centers):
                    xn, yn = region
                    d = region_distance_functions[i](x, y, xn, yn)

                    if d < d_min:
                        d_min = d

                        if d <= d_limit:
                            image[x][y] = id(region)

class Utilities:
    def error(message, q=True):
        print(f"\u001b[38;5;1mERROR:\u001b[0m {message}", flush=True)

        if q:
            sys.exit(1)

    def warning(message):
        print(f"\u001b[38;5;208mWARNING:\u001b[0m {message}", flush=True)

    def info(message):
        print(f"\u001b[38;5;11mINFO:\u001b[0m {message}", flush=True)

    def success(message):
        print(f"\u001b[38;5;2mSUCCESS:\u001b[0m {message}", flush=True)

    def hex_to_tuple(color: str):
        color = color.strip("#")
        return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))

    def get_different_adjacent_colors(width, height, image, colors, color_algorithm):
        from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD

        edges = set()

        mapping = {}
        n = 0

        for x in range(width):
            for y in range(height):
                for xd, yd in ((0, 1), (1, 0), (-1, 0), (0, -1)):
                    xn, yn = x + xd, y + yd

                    if not 0 <= xn < width or not 0 <= yn < height:
                        continue

                    i1, i2 = image[x][y], image[xn][yn]

                    if i1 is None or i2 is None:
                        continue

                    if i1 < i2:
                        if i1 not in mapping:
                            n += 1
                            mapping[n] = i1
                            mapping[i1] = n

                        if i2 not in mapping:
                            n += 1
                            mapping[n] = i2
                            mapping[i2] = n

                        edges.add((mapping[i1], mapping[i2]))

        edges = list(edges)
        model = LpProblem(sense=LpMinimize)

        chromatic_number = LpVariable(name="chromatic number", cat='Integer')
        variables = [[LpVariable(name=f"x_{i}_{j}", cat='Binary') \
                      for i in range(n)] for j in range(n)]

        for i in range(n):
            model += lpSum(variables[i]) == 1
        for u, v in edges:
            for color in range(n):
                model += variables[u - 1][color] + variables[v - 1][color] <= 1
        for i in range(n):
            for j in range(n):
                model += chromatic_number >= (j + 1) * variables[i][j]

        if color_algorithm == ColorAlgorithm.least_possible:
            model += chromatic_number
        else:
            model += chromatic_number == len(colors)

        status = model.solve(PULP_CBC_CMD(msg=False))

        if chromatic_number.value() > len(colors):
            Utilities.error("Not enough colors to color without adjacent areas having the same one!")

        return {mapping[variable + 1]: colors[color]
                for variable in range(n)
                for color in range(n)
                if variables[variable][color].value() == 1}

    def add_border(background, border_size, read_image, write_image, width, height, mask_function):
        r = border_size // 2

        for x in range(width):
            for y in range(height):
                if not mask_function((x, y)):
                    continue

                for dx, dy in ((0, 1), (1, 0)):
                    xn, yn = x + dx, y + dy

                    if not 0 <= xn < width or not 0 <= yn < height:
                        continue

                    if not mask_function((xn, yn)):
                        continue

                    if read_image[x][y] != read_image[xn][yn]:
                        draw = ImageDraw.Draw(write_image)
                        draw.ellipse((x-r, y-r, x+r, y+r), fill=(*background,0))


def generate(       
        regions: int,
        colors: List[Union[Tuple[int], str]],
        width: int = 1920,
        height: int = 1080,
        region_algorithm = RegionAlgorithm.uniform,
        distance_algorithm = DistanceAlgorithm.euclidean,
        color_algorithm = ColorAlgorithm.random,
        seed: Optional[int] = None,
        border_size: int = 0,
        mask: Optional[str] = None,
        mask_color = "#000000",
        animate = False,
        background = "#FFFFFF",
):
    # possibly seed the random algorithm
    if seed is None:
        seed = random()

    # possibly convert string colors to tuples
    i = 0
    while i < len(colors):
        if type(colors[i]) == str:
            colors[i] = Utilities.hex_to_tuple(colors[i])

        i += 1

    if type(mask_color) == str:
        mask_color = Utilities.hex_to_tuple(mask_color)
    elif type(mask_color) == list:
        mask_color = tuple(mask_color)

    if type(background) == str:
        background = Utilities.hex_to_tuple(background)
    elif type(background) == list:
        background = tuple(background)

    randomseed(seed)

    mask_function = lambda p: True

    if mask is not None:
        try:
            mask_img = Image.open(mask)
            Utilities.info("Mask provided.")

            w, h = mask_img.size

            mask_function = lambda p: mask_img.getpixel(p) == mask_color

            if w != width:
                Utilities.warning("Specified width doesn't match mask width, using mask width.")
                width = w

            if h != height:
                Utilities.warning("Specified height doesn't match mask height, using mask width.")
                height = h


        except Exception as e:
            Utilities.error(f"Error loading mask from '{mask}'.")

    if type(regions) == list:
        Utilities.info("Region centers provided, skipping generation.")

        # flip vertically!
        region_centers = [(int(center[0] * width), int(height - center[1] * height)) for center in regions]
    else:
        Utilities.info("Calculating region centers.")
        region_centers = region_algorithm(width, height, regions, mask_function)

    image = [[None] * height for _ in range(width)]
    Utilities.info("Calculating region areas.")
    DistanceAlgorithm.set_each_point(seed, width, height, region_centers, image, float("inf"), distance_algorithm, mask_function)

    # either assign colors randomly, or calculate the chromatic number and assign them then
    if color_algorithm == ColorAlgorithm.random:
        Utilities.info("Assigning region colors.")
        region_colors = {id(region): choice(colors) for region in region_centers}
    else:
        Utilities.info("Assigning region colors such that no two adjacent regions have the same color.")
        region_colors = Utilities.get_different_adjacent_colors(width, height, image, colors, color_algorithm)

    # if we're masking, some regions won't be assigned
    region_colors[None] = background

    # the original, full image (without borders)
    pil_image = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            pil_image.putpixel((x, y), region_colors[image[x][y]])

    if border_size != 0:
        Utilities.add_border(background, border_size, image, pil_image, width, height, mask_function)

    if animate:
        if not os.path.exists(path):
            os.makedirs(path)

        d = 1

        while True:
            animation_image = [[None] * height for _ in range(width)]
            DistanceAlgorithm.set_each_point(seed, width, height, region_centers, animation_image, d, distance_algorithm, mask_function)

            animation_pil_image = Image.new("RGB", (width, height))

            for x in range(width):
                for y in range(height):
                    animation_pil_image.putpixel((x, y), background if animation_image[x][y] is None else region_colors[image[x][y]])

            if border_size != 0:
                Utilities.add_border(background, border_size, animation_image, animation_pil_image, width, height)

            animation_path = os.path.join(path, f"{d}.png")

            animation_pil_image.save(animation_path, "PNG")
            Utilities.success(f"Animation image saved to {animation_path}")

            d += 1

            if image == animation_image:
                Utilities.success(f"Done!")
                break

    else:        
        image_array = np.array(pil_image)
        return image_array
