import warnings
warnings.filterwarnings('ignore')

import numpy as np
from skimage import transform, io

from julesz import load_lib, get_histogram, julesz
from filters import get_filters

import copy
import matplotlib as plt

def main():
    # TODO consider adjusting this value
    max_intensity = 7

    lib = load_lib()

    # (1) generate filter bank
    [F, filters, width, height] = get_filters()
    im_w = im_h = 256

    # # Load all images
    images = []
    for img_name in ['fur_obs.jpg', 'grass_obs.bmp', 'stucco.bmp']:
        img = io.imread(f'images/{img_name}', as_grey=True)
        img = transform.resize(img, (im_w, im_h), mode='symmetric', preserve_range=True)
        if img_name == 'grass_obs.bmp':
            img = (img / 255 * max_intensity).astype(np.int32)
        else:
            img = (img * max_intensity).astype(np.int32)
        images.append(img)


    err_threshold = 0
    bin_weights = np.array([8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8])

    image_num = 0

    for image in images:
        image_num += 1

        # get_histogram = filtered image (response)

        chosen_filters = []
        chosen_width = []
        chosen_height = []
        unselected_filters = range(len(filters))
        synth_errors_1 = []
        synth_errors_2 = []

        synth_image = np.random.randint(0, 8, size=(im_w, im_h)).astype(np.int32)
        # synth_image = np.rint(np.random.uniform(0, 7, (im_w, im_h))).astype(np.int32)
        print("================== Image ==================")

        # while the filter bank is not empty
        while len(unselected_filters) > 0:

            selected_filter_num = len(filters) - len(unselected_filters)

            filter_errs = []
            # calculate the filter response for each unselected filter
            for filter_num in unselected_filters:
                h1 = get_histogram(lib, image, filters[[filter_num], :], width[[filter_num]], height[[filter_num]], max_intensity=max_intensity)

                h2 = get_histogram(lib, synth_image, filters[[filter_num], :], width[[filter_num]], height[[filter_num]], max_intensity=max_intensity)

                # TODO: for the errors, match the tails more closely than the centers
                # should change code in jules.c with the stuff from 3b
                err = np.dot(np.abs(h1 - h2).flatten(), bin_weights)
                filter_errs.append(err)

            filter_idx = np.argmax(filter_errs)
            if filter_errs[filter_idx] < err_threshold:
                break

            # remove filter once it's selected
            chosen_filters.append(unselected_filters[filter_idx])
            unselected_filters = np.delete(unselected_filters, filter_idx, 0)

            print(f"{chosen_filters} {unselected_filters}")

            # synthesize new image
            h = get_histogram(lib, image, filters[chosen_filters, :], width[chosen_filters], height[chosen_filters], max_intensity=max_intensity)

            synth_image, synth_resp = julesz(lib, h.reshape(len(chosen_filters), -1), filters[chosen_filters, :], width[chosen_filters], height[chosen_filters], im_w, im_h, max_intensity=max_intensity)
            synth_image = synth_image.reshape((im_w, im_h))

            print(f"Filter {selected_filter_num} uses idx: {filter_idx}")
            print(f"  Img diff = {np.linalg.norm(synth_image - image)}")
            print(f"  Hist err = {np.linalg.norm(h - synth_resp)}")
            print(f"  avg bin err = {np.mean(np.dot(np.abs(h - synth_resp).reshape(len(chosen_filters), -1), bin_weights)/15)}")
            io.imsave(f'output/synth_{image_num}_{selected_filter_num}.png', (synth_image*32).astype(np.uint))

            synth_errors_1.append(np.linalg.norm(synth_image - image))
            synth_errors_2.append(np.mean(np.dot(np.abs(h - synth_resp).reshape(len(chosen_filters), -1), bin_weights)/15))

            with open(f"output/errors_{image_num}.txt", "a") as f:
                f.write(f"{selected_filter_num}, {np.linalg.norm(synth_image - image)}, {np.linalg.norm(h - synth_resp)}, {np.mean(np.dot(np.abs(h - synth_resp).reshape(len(chosen_filters), -1), bin_weights)/15)}\n")

            # stop after 20 filters
            if selected_filter_num > 23:
                break

        # record errors
        with open(f"output/finalerrors1_{image_num}.txt", "w") as f:
            for err in synth_errors_1:
                f.write(str(err) + "\n")
        with open(f"output/finalerrors2_{image_num}.txt", "w") as f:
            for err in synth_errors_2:
                f.write(str(err) + "\n")
        break

        # record which filters were chosen
        with open(f"chosen_filters_{image_num}.txt", "w") as f:
            f.write(str(chosen_filters))

    print("====== Done ===================")

    # Save the picture of the filters used
    # fig, axes = plt.subplots(8, 6, figsize=(12,18))
    # plt.subplots_adjust(hspace = 0.5)
    # for i in range(len(F)):
    #     r = int(i / 6)
    #     c = i % 6
    #     axes[r, c].imshow(F[i], cmap='gist_gray')
    #     axes[r, c].set_title(f"Filter {i}")
    # axes[7, 5].axis('off')
    # plt.savefig("filters.png")


if __name__ == '__main__':
    main()


'''
You may want to (i) change the computation of 'mult' such that the range of each filter response is utilized, and, (ii) adjust 'halfnum' such that non-zero centered histograms are supported (e.g. convolution with identity kernel).

what is the bin range in part 2 referring to?
by GONG, STEVEN - Monday, 12 November 2018, 1:39 AM PST
Is it the size of each bin or the number of bins? Why do we need to specify that? In the provided code,  there is a specification that we need to compute the bin width according to the maximum and minimum of the filter response. Does it mean that we want to have a fixed number of bins?

Re: what is the bin range in part 2 referring to?
by NIJKAMP, ERIK - Tuesday, 13 November 2018, 9:47 AM PST
Pass in the minimum / maximum filter response and adjust the bins accordingly. The number of bins is fixed.


'''
