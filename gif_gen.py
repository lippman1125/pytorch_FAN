import imageio
import os
import os.path
import cv2

def create_gif(gif_name, path, duration = 0.3):
    '''
    生成gif文件，原始图片仅支持png格式
    gif_name ： 字符串，所生成的 gif 文件名，带 .gif 后缀
    path :      需要合成为 gif 的图片所在路径
    duration :  gif 图像时间间隔
    '''

    frames = []
    pngFiles = os.listdir(path)
    image_list = [os.path.join(path, f) for f in pngFiles]
    image_list.sort(key= lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))

    for image_name in image_list:
        print(image_name)
        # 读取 png 图像文件
        img = cv2.imread(image_name)
        img_crop = img[200:, 200:, :]
        img_resize = cv2.resize(img_crop, (180, 120))

        frames.append(img_resize[:,:,::-1])
    # 保存为 gif
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)

    return

def main():
    gif_name = 'landmarks.gif'
    path = './save_pics'   #指定文件路径
    duration = 0.05
    create_gif(gif_name, path, duration)

if __name__ == "__main__":
    main()
