from os import listdir
import SimpleITK
import os.path
import pickle
import numpy
import matplotlib.pyplot as plt
import SimpleITK as sitk

volSize = numpy.array((256,256,32), numpy.int32)
# dstRes  = numpy.array((1,1,1.5))
normDir = False
method  = SimpleITK.sitkLinear

def process_scan(scan):
    ret = numpy.zeros(volSize, dtype=numpy.float32)
    newSize = numpy.array([256, 256, 32])
    factor = numpy.asarray(scan.GetSize()) / newSize
    new_spacing = numpy.asarray(scan.GetSpacing()) * factor

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(scan)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(newSize.tolist())
    resampler.SetInterpolator(method)

    imgResampled = resampler.Execute(scan)

    # a = numpy.zeros([128, 128, 64], dtype=numpy.float32)
    # a = numpy.transpose(sitk.GetArrayFromImage(scan).astype(dtype=float), [2, 1, 0])
    # plt.imshow(a[:, :, 0])
    # plt.show()
    # ret = numpy.zeros(volSize, dtype=numpy.float32)
    # factor = numpy.asarray(scan.GetSpacing()) / dstRes
    #
    # factorSize = numpy.asarray(scan.GetSize() * factor, dtype=numpy.float)
    #
    # newSize = numpy.max([factorSize, volSize], axis=0)
    #
    # newSize = newSize.astype(dtype=numpy.int32)
    #
    # T=SimpleITK.AffineTransform(3)
    # T.SetMatrix(scan.GetDirection())
    #
    # resampler = SimpleITK.ResampleImageFilter()
    # resampler.SetReferenceImage(scan)
    # resampler.SetOutputSpacing(dstRes)
    # resampler.SetSize(newSize.tolist())
    # resampler.SetInterpolator(method)
    # if normDir:
    #     resampler.SetTransform(T.GetInverse())
    #
    # imgResampled = resampler.Execute(scan)


    # imgCentroid = numpy.asarray(newSize, dtype=numpy.float) / 2.0
    #
    # imgStartPx = (imgCentroid - numpy.array(volSize) / 2.0).astype(dtype=int)
    #
    # regionExtractor = SimpleITK.RegionOfInterestImageFilter()
    # regionExtractor.SetSize(volSize.astype(dtype=numpy.int32).tolist())
    # regionExtractor.SetIndex(imgStartPx.tolist())
    #
    # imgResampledCropped = regionExtractor.Execute(imgResampled)

    return numpy.transpose(
        SimpleITK.GetArrayFromImage(imgResampled).astype(dtype=numpy.float),
        [2, 1, 0]
    )




def iterate_folder(folder):
    for filename in sorted(listdir(folder)):
        if filename.endswith('.mhd'):
            absolute_filename = os.path.join(folder, filename)

            yield absolute_filename




def load_data(folder):
    input_filenames = list(iterate_folder(folder))

    print(input_filenames)

    X = numpy.array([process_scan(SimpleITK.ReadImage(f)) for f in input_filenames])   #(50.128.128.160)

    # print(X.shape)
    # print(X.shape)

    # plt.imshow(X[0,:,:,10])
    # plt.show()

    return X


X= load_data('I:\P3data\mhd1-2-3-4\\')
#X.shape
# plt.imshow(y[0,:,:,10])
# plt.show()
with open('I:\P3data\mhd1-2-3-4\\train_without5.p3', 'wb') as f:
    pickle.dump(X, f)

