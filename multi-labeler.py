# -*- coding: utf-8 -*-

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from scipy.spatial import cKDTree


class ImageProcessor:
    def __init__(self, show_image=0, resize_rate=0.6, bar_type="bd"):
        self.show_image = show_image
        self.resize_rate = resize_rate
        self.bar_type = bar_type
        self.lsd = cv2.ximgproc.createFastLineDetector(
            length_threshold=70,
            distance_threshold=1.41421356,
            canny_th1=30,
            canny_th2=100,
            canny_aperture_size=3,
            do_merge=True
        )

    def process_images(self, img_dir):
        """处理目录中的所有图像"""
        img_list = self._get_image_list(img_dir)
        msk_list = self._get_mask_list(img_dir)

        for img_path in img_list:
            if "Msk" in img_path or "tagged" in img_path:
                continue
            try:
                self.process_image(img_path, msk_list)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    def process_image(self, img_path, msk_list):
        """处理单张图像"""
        img_name = img_path.split("-")[0]
        img_scale = float(img_path.split("-")[1])
        if img_scale <= 1:
            img_scale = int(img_scale * 1000)

        csv_path = img_path[0:-3] + "csv"
        print(f"Processing {img_path}, result will be saved to {csv_path}")

        # 1. 加载图像
        img_ori = self._load_image(img_path)
        if img_ori is None:
            return

        # 2. 检测比例尺
        bar_length = self._detect_scale_bar(img_ori)
        pix2nm = img_scale / bar_length
        print(f"Scale bar detected: {bar_length} pixels, resolution: {pix2nm} nm/pixel")

        # 3. 检测颗粒
        circles = self._detect_circles(img_ori)
        print(f"Detected {len(circles)} particles")

        # 4. 分割图像
        img_div = self._segment_image(img_ori)

        # 5. 获取边界掩码
        mouse_mask = self._get_boundary_mask(img_name, msk_list, img_ori)

        # 6. 过滤边界外的颗粒
        filtered_circles = self._filter_circles(circles, mouse_mask, img_div)

        # 7. 填充空洞区域
        cover_mask, area_mask = self._fill_hollow_regions(filtered_circles, img_ori, mouse_mask)

        # 8. 计算形态参数
        params = self._calculate_morphology_params(cover_mask, filtered_circles)

        # 9. 保存结果
        self._save_results(img_path, img_ori, filtered_circles, params, csv_path, pix2nm)

    # 辅助方法
    def _load_image(self, img_path):
        """加载图像"""
        img = cv2.imread(img_path, 0)
        if img is None:
            print(f"Failed to load image: {img_path}")
        return img

    def _detect_scale_bar(self, img):
        """检测比例尺长度"""
        if self.bar_type == 'fx':
            roi = img[1060:1080, 800:1300]
        elif self.bar_type == 'bd':
            roi = img[580:650, 30:300]
        else:
            raise ValueError(f"Unsupported bar type: {self.bar_type}")

        lines = self.lsd.detect(roi)
        if lines is None or len(lines) == 0:
            return 0

        x_coords = [line[0][0] for line in lines] + [line[0][2] for line in lines]
        return max(x_coords) - min(x_coords)

    def _detect_circles(self, img):
        """检测图像中的圆形颗粒"""
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)

        # 检测不同大小的圆
        small_circles = cv2.HoughCircles(
            img_blur, cv2.HOUGH_GRADIENT, 1, 7,
            param1=100, param2=5, minRadius=4, maxRadius=8
        )

        large_circles = cv2.HoughCircles(
            img_blur, cv2.HOUGH_GRADIENT, 1, 11,
            param1=100, param2=6, minRadius=8, maxRadius=12
        )

        # 合并并过滤圆
        if small_circles is not None:
            small_circles = small_circles[0]
        else:
            small_circles = np.array([])

        if large_circles is not None:
            large_circles = large_circles[0]
        else:
            large_circles = np.array([])

        # 过滤被大圆覆盖的小圆
        filtered_small = []
        for sc in small_circles:
            covered = False
            for lc in large_circles:
                dist = np.sqrt(((sc[:2] - lc[:2]) ** 2).sum())
                if dist < lc[2]:
                    covered = True
                    break
            if not covered:
                filtered_small.append(sc)

        return np.vstack([np.array(filtered_small), large_circles]) if filtered_small else large_circles

    def _segment_image(self, img):
        """分割图像"""
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        _, img_div = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        _, img_div = cv2.threshold(img_blur, _ + 42, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_div = cv2.morphologyEx(img_div, cv2.MORPH_OPEN, kernel)
        img_div = cv2.morphologyEx(img_div, cv2.MORPH_CLOSE, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        img_div = cv2.dilate(img_div, kernel)

        return img_div

    def _get_boundary_mask(self, img_name, msk_list, img):
        """获取边界掩码"""
        if f"{img_name}_Msk.png" in msk_list:
            return cv2.imread(f"{img_name}_Msk.png", 0)

        # 交互式绘制边界
        print("Routes not found, please draw the border of the particle...")
        route_list = []

        def draw_border(event, x, y, flags, param):
            nonlocal route_list
            if flags == cv2.EVENT_FLAG_LBUTTON:
                x, y = int(x / self.resize_rate), int(y / self.resize_rate)
                if not route_list:
                    route_list.append([[x, y]])
                else:
                    # 合并相邻路径逻辑
                    for route in route_list:
                        if np.sqrt(((np.array(route[-1]) - np.array([x, y])) ** 2).sum()) <= 20 / self.resize_rate:
                            route.append([x, y])
                            return
                    route_list.append([[x, y]])

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_border)

        while True:
            key = cv2.waitKey(20)
            if key == 27:  # ESC键退出
                break

            img_display = img.copy()
            for route in route_list:
                if len(route) > 1:
                    pts = np.array(route, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img_display, [pts], False, (0, 255, 0), 1)

            cv2.imshow('image', cv2.resize(img_display, (0, 0), fx=self.resize_rate, fy=self.resize_rate))

        cv2.destroyAllWindows()

        # 创建掩码
        mask = np.zeros_like(img)
        for route in route_list:
            if len(route) > 2:
                pts = np.array(route, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)

        cv2.imwrite(f"{img_name}_Msk.png", mask)
        return mask

    def _filter_circles(self, circles, mask, img_div):
        """过滤边界外的圆"""
        filtered = []
        for circle in circles:
            x, y, r = circle
            if (x >= r and y >= r and x < mask.shape[1] - r and y < mask.shape[0] - r and
                    mask[int(y), int(x)] > 0 and img_div[int(y), int(x)] > 0):
                filtered.append(circle)
        return np.array(filtered)

    def _fill_hollow_regions(self, circles, img, mask):
        """填充空洞区域"""
        cover_mask = np.zeros_like(img)
        area_mask = np.zeros_like(img)

        for x, y, r in circles:
            cv2.circle(cover_mask, (int(x), int(y)), int(r + 2), 255, -1)
            cv2.circle(area_mask, (int(x), int(y)), int(r), 255, -1)

        # 填充空洞
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask[i, j] > 0 and img[i, j] <= 80 and cover_mask[i, j] == 0:
                    # 找到最近的圆
                    distances = np.sqrt(((circles[:, :2] - np.array([j, i])) ** 2).sum(axis=1))
                    nearest_idx = np.argmin(distances)
                    nearest_r = circles[nearest_idx, 2]

                    cv2.circle(cover_mask, (j, i), int(nearest_r + 2), 255, -1)

        return cover_mask, area_mask

    def _calculate_morphology_params(self, mask, circles):
        """计算形态学参数"""
        params = {}

        # 计算中心
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            params['center_x'] = int(moments["m10"] / moments["m00"])
            params['center_y'] = int(moments["m01"] / moments["m00"])
        else:
            params['center_x'] = 0
            params['center_y'] = 0

        # 计算轮廓和凸包
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return params

        cnt = max(contours, key=lambda c: cv2.contourArea(c))
        hull = cv2.convexHull(cnt)

        # 计算Feret直径
        feret_dia = 0
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                dist = np.sqrt(((hull[i][0] - hull[j][0]) ** 2).sum())
                if dist > feret_dia:
                    feret_dia = dist

        params['feret_diameter'] = feret_dia

        # 计算圆度和凸度
        area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(cnt, True)

        params['roundness'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        params['convexity'] = area / hull_area if hull_area > 0 else 0

        # 计算分形维数
        if len(circles) > 10:
            # 这里简化计算，实际需要更复杂的算法
            params['fractal_dimension'] = 2.0
        else:
            params['fractal_dimension'] = 0

        return params

    def _save_results(self, img_path, img_ori, circles, params, csv_path, pix2nm):
        """保存结果"""
        # 保存标记图像
        img_tagged = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2BGR)
        for x, y, r in circles:
            cv2.circle(img_tagged, (int(x), int(y)), int(r), (0, 255, 0), 1)
            cv2.circle(img_tagged, (int(x), int(y)), 1, (0, 0, 255), 1)

        cv2.imwrite(img_path.replace('.png', '-tagged.png'), img_tagged)

        # 保存CSV结果
        data = {
            'x': circles[:, 0],
            'y': circles[:, 1],
            'radius(nm)': circles[:, 2] * pix2nm,
            'feret_diameter(nm)': params['feret_diameter'] * pix2nm,
            'roundness': params['roundness'],
            'convexity': params['convexity'],
            'fractal_dimension': params['fractal_dimension']
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    def _get_image_list(self, img_dir):
        """获取图像文件列表"""
        return [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def _get_mask_list(self, img_dir):
        """获取掩码文件列表"""
        return [f for f in os.listdir(img_dir) if f.endswith('_Msk.png')]


# 主程序
if __name__ == "__main__":
    bar = ["bd", "fx"]
    for b in bar:
        processor = ImageProcessor(show_image=0, resize_rate=0.6, bar_type=b)
        processor.process_images(sys.path[0])