# Generates different types of data

import numpy as np
import matplotlib.pyplot as plt

class DataGenerator(object):

    def _generate_rand_xs(self, dimensions):
        return np.append(np.random.randint(0, 10, (dimensions-1, ))*np.random.choice([1, -1]), 1)

    def _get_data_points(self, data, count, dimensions, coefficients, number_of_datapoints, _class):
        for data_point in range(count):
            x_s = self._generate_rand_xs(dimensions)
            dot_product = np.dot(x_s, coefficients)
            if _class != 'positive':
                shifted_data = x_s[:-1] + np.random.randint(-number_of_datapoints, -2)
            else:
                shifted_data = x_s[:-1] + np.random.randint(2, number_of_datapoints)
            single_data = np.append(shifted_data, dot_product)
            data.append(single_data.tolist())
        return data

    # def linear(self, slope, variane, dimensions = 2):

    def triangle_rect_images(self, number_of_images, save_as_image):
        img_size = 16
        min_rect_size = 3
        max_rect_size = 8
        num_objects = 2

        if not save_as_image:
            data = []

        number_of_rects = number_of_images // 2
        number_of_triangles = number_of_images - number_of_rects
        for i_img in range(number_of_rects):
            imgs = np.zeros((img_size, img_size))
            width, height = np.random.randint(min_rect_size, max_rect_size, size=2)
            x = np.random.randint(0, img_size - width)
            y = np.random.randint(0, img_size - height)
            imgs[x:x+width, y:y+height] = 1.
            if not save_as_image:
                data.append(imgs.T)
            else:
                plt.imsave('square'+str(i_img)+'.jpeg', imgs.T)
        
        for i_img in range(number_of_triangles):
            imgs = np.zeros((img_size, img_size))
            size = np.random.randint(min_rect_size, max_rect_size)
            x, y = np.random.randint(0, img_size - size, size=2)
            mask = np.tril_indices(size)
            imgs[x + mask[0], y + mask[1]] = 1.
            if not save_as_image:
                data.append(imgs.T)
            else:
                plt.imsave('triangle'+str(i_img)+'.jpeg', imgs.T)

        if not save_as_image:
            return data

    def linearly_separable (self, number_of_datapoints, coefficients, dimensions = 2):

        if dimensions != len(coefficients):
            raise Exception('Dimensions and coefficient array should have same length.')

        positive_class_count = number_of_datapoints // 2
        negative_class_count = number_of_datapoints - positive_class_count

        #convert coefficient array to numpy
        coefficients = np.array(coefficients)
        data = self._get_data_points([], positive_class_count, dimensions, coefficients, number_of_datapoints, 'positive')
        return self._get_data_points(data, negative_class_count, dimensions, coefficients, number_of_datapoints, 'negative')


data_gen = DataGenerator()
print (data_gen.triangle_rect_images(2, save_as_image = True))

