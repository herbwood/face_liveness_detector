import os
import shutil
import glob
import zipfile
from utils.utils import configInfo

# 400명 인당 100
# sl : 악세서리 착용 여부(s1, 2)
# ll : 조도(l1, 2, 3, 8, 9)
# el : 표정(e1, 2)
# cl : 포즈 방향(c4, 7, 10, 19, 20)장

# def sampleTrainDataset(config, basepath='temp', targetpath='whole'):
#     # sl, ll, el, cl을 조합하여 우리가 추출할 이미지 데이터 path를 지정해줌
#     config = configInfo(config)
#     sl, ll, el, cl = config["data_gatherer"]
#
#     # zip 파일을 풀면 그냥 생짜로 S001~S006 디렉터리를 풀어버려서 임시로 보관할 디렉터리가 필요해서
#     # temp 디렉터리 안에 압축을 해제함
#     # whole은 추출한 이미지 데이터만 저장하는 디렉터리
#     # directory 변수에 zipfile이 저장된 디렉터리로 지정해주고
#     # 데이터셋 zipfile말고 다른 zipfile은 없개끔!!!!!!!!!!!!
#     # 저처럼 그냥 zipfile이 저장된 디렉터리에서 실행시킬거면 directory 안바꾸셔도 되요
#     directory = './'
#     os.chdir(directory)
#
#     # temp 디렉터리 만들어주기
#     if basepath not in os.listdir():
#         os.mkdir(basepath)
#
#     # whole 디렉터리 만들어주기
#     if targetpath not in os.listdir():
#         os.mkdir(targetpath)
#
#     # 디렉터리 내에 있는 모든 zip 파일을
#     for zf in glob.glob('*.zip'):
#         pathlist = []
#         zipname = zf.split('.')[0]  # zip 파일 이름
#
#         # zip 파일을 temp 디렉터리에 압축 해제해줌
#         zf = zipfile.ZipFile(zf)
#         zf.extractall(os.path.join(basepath))
#
#         # (추출한 이미지 파일의 경로, 새롭게 지정할 이미지 파일의 이름) 형식으로 pathlist 변수에 저장함
#         for s in sl:
#             for l in ll:
#                 for e in el:
#                     for c in cl:
#                         pathlist.append((os.path.join(s, l, e, c), '_'.join([zipname, s, l, e, c])))
#
#         # temp 디렉터리에 저장한 추출한 이미지 파일을 whole 디렉터리에 옮겨줌
#         for path, pname in pathlist:
#             shutil.move(os.path.join(basepath, path), os.path.join(targetpath, pname))
#             print(path, pname)
#
#         # temp 디렉터리에 있는 쓰지 않는 디렉터리 및 파일 제거
#         for dir in os.listdir(basepath):
#             dpath = os.path.join(basepath, dir)
#             if os.path.isdir(dpath):
#                 shutil.rmtree(dpath)
#
#     # zip 파일을 전부 다 압축해제하면 temp 파일도 지워주기
#     shutil.rmtree(basepath)

def extract_image_name(basepath, file):
    with open(file) as f:
        extracted = f.readlines()

    extracted = [os.path.join(basepath, data.split(' ')[0].split('\\')[0], data.split(' ')[0].split('\\')[1]) for data
                 in extracted]

    return extracted


def copy_images_to_dir(imagepathlist, target_dir):
    for imagepath in imagepathlist:
        filename = os.path.basename(imagepath)
        shutil.copy(imagepath, os.path.join(target_dir, filename))


# if __name__ == "__main__":
#     config = configInfo("../config.json")
    # metadata = config["data_gatherer"]
    # sl, ll, el, cl = metadata["sl"], metadata["ll"], metadata["el"], metadata["cl"]
    # print(sl, ll, el, cl)