import OpenEXR
import Imath
import numpy as np
from PIL import Image
def arr2np(arr):
    channels = len(arr)
    height, width = arr[0].shape
    x = np.zeros((height, width, channels))
    for c in range(channels):
        x[:, :, c] = arr[c]
    x = np.abs(x) ** 0.5
    x = np.clip(x,0,1)
    return x

eps = 0.00316
def divide(d_rgb,d_albedo):
    # contains_zero = np.any(d_albedo == 0)
    # if contains_zero:
    #     print("数组中包含零")
    # else:
    #     print("数组中不包含零")
    # d_albedo对应位置= 0时，结果也为0
    # out = np.where(d_albedo == 0, 0, np.divide(d_rgb, d_albedo))
    # 对应位置<0.001时，结果为0
    out = np.where(d_albedo < 0.001, 0, np.divide(d_rgb, d_albedo))
    # 这段的意思是，只有d_albedo对应位置！ = 0时，才会将两者相除
   # out = np.divide(d_rgb,d_albedo,out=d_albedo,where=d_albedo != 0)

    # out = np.divide(d_rgb, d_albedo , out=d_rgb)
    return out


def get_illu(filename):
    # 只需要6个通道：illu_B,illu_G,ill_R,"Sdf_B","Sdf_G","Sdf_R"
    # 每个通道要除以alpha，如果alpha=0不用除

    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()

    channel_names = ["Alpha", "SingleRadiDividedBySdf_B", "SingleRadiDividedBySdf_G", "SingleRadiDividedBySdf_R"]
    channel_datas = []
    for channelname in channel_names:
        half_channel = exr_file.channel(channelname, Imath.PixelType(Imath.PixelType.HALF))
        width = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
        height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1
        half_array = np.frombuffer(half_channel, dtype=np.float16)
        half_array = np.reshape(half_array, (height, width))

        channel_datas.append(half_array)
    channel_datas = np.array(channel_datas)

    I_appro = np.stack((channel_datas[1], channel_datas[2], channel_datas[3]), axis=2)
    # Radi = np.stack((channel_datas[4], channel_datas[5], channel_datas[6]), axis=2)
    # Alpha = np.stack((channel_datas[0], channel_datas[0], channel_datas[0]), axis=2)



    # Illu_appro = Illu_appro*Alpha
    # Illu_noalpha = divide(Illu, Alpha)
    # output = np.stack((Sdf_noalpha, Illumin_noalpha), axis=2)
    return I_appro



def get_sdf(filename):
    # 只需要6个通道：illu_B,illu_G,ill_R,"Sdf_B","Sdf_G","Sdf_R"
    # 每个通道要除以alpha，如果alpha=0不用除

    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()

    channel_names = ["Alpha", "Sdf_B", "Sdf_G", "Sdf_R"]
    channel_datas = []
    for channelname in channel_names:
        half_channel = exr_file.channel(channelname, Imath.PixelType(Imath.PixelType.HALF))
        width = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
        height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1
        half_array = np.frombuffer(half_channel, dtype=np.float16)
        half_array = np.reshape(half_array, (height, width))

        channel_datas.append(half_array)
    channel_datas = np.array(channel_datas)

    Sdf = np.stack((channel_datas[1], channel_datas[2], channel_datas[3]), axis=2)
    # Alpha = np.stack((channel_datas[0], channel_datas[0], channel_datas[0]), axis=2)

    # output = np.stack((Sdf_noalpha, Illumin_noalpha), axis=2)
    return Sdf
# 直接得到radiance no alpha data
def get_volumn_radiance(filename):
    # 只需要6个通道：illu_B,illu_G,ill_R,"Sdf_B","Sdf_G","Sdf_R"
    # 每个通道要除以alpha，如果alpha=0不用除

    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()

    channel_names = ["Alpha","SingleRadi_B", "SingleRadi_G", "SingleRadi_R"]
    channel_datas = []
    for channelname in channel_names:
        half_channel = exr_file.channel(channelname, Imath.PixelType(Imath.PixelType.HALF))
        width = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
        height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1
        half_array = np.frombuffer(half_channel, dtype=np.float16)
        half_array = np.reshape(half_array, (height, width))

        channel_datas.append(half_array)
    channel_datas = np.array(channel_datas)

    Radiance = np.stack((channel_datas[1], channel_datas[2], channel_datas[3]), axis=2)
    Alpha = np.stack((channel_datas[0], channel_datas[0], channel_datas[0]), axis=2)

    # Rad_noalpha = divide(Radiance, Alpha)

    # output = np.stack((Sdf_noalpha, Illumin_noalpha), axis=2)
    return Radiance
