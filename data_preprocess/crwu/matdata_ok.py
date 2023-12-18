"""
@project:Bearing_Fault_Diagnosis
@Author: Phantom
@Time:2023/12/17 下午3:31
@Email: 2909981736@qq.com
"""
matdata = {'12kDE': [{'class': 0, 'classname': 'normal', 'faultname': '基座,健康,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_3.mat']
                      },
                     {'class': 1, 'classname': 'IR007', 'faultname': '驱动端,内圈,故障等级1,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0007/IR007_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0007/IR007_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0007/IR007_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0007/IR007_3.mat']
                      },
                     {'class': 2, 'classname': 'B007', 'faultname': '驱动端,滚珠圈,故障等级1,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0007/B007_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0007/B007_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0007/B007_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0007/B007_3.mat']
                      },

                     {'class': 3, 'classname': 'OR007', 'faultname': '驱动端,外圈,故障等级1,位置@6-3-12,载荷0-3',
                      'srcurl': [
                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_0.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Orthogonal/0007/144.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_0.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_1.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Orthogonal/0007/145.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_1.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_2.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Orthogonal/0007/146.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_2.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_3.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Orthogonal/0007/147.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_3.mat']]
                      },

                     {'class': 4, 'classname': 'IR014', 'faultname': '驱动端,内圈,故障等级2,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0014/IR014_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0014/IR014_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0014/IR014_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0014/IR014_3.mat']
                      },

                     {'class': 5, 'classname': 'B014', 'faultname': '驱动端,滚珠圈,故障等级2,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0014/B014_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0014/B014_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0014/B014_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0014/B014_3.mat']
                      },

                     {'class': 6, 'classname': 'OR014', 'faultname': '驱动端,外圈,故障等级2,位置@6,载荷0-3',
                      'srcurl': [[
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0014/OR014@6_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0014/OR014@6_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0014/OR014@6_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0014/OR014@6_3.mat']]
                      },

                     {'class': 7, 'classname': 'IR021', 'faultname': '驱动端,内圈,故障等级3,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0021/IR021_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0021/IR021_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0021/IR021_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Inner Race/0021/IR021_3.mat']
                      },

                     {'class': 8, 'classname': 'B021', 'faultname': '驱动端,滚珠圈,故障等级3,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0021/B021_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0021/B021_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0021/B021_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Ball/0021/B021_3.mat']
                      },

                     {'class': 9, 'classname': 'OR021', 'faultname': '驱动端,外圈,故障等级3,位置@6-3-12,载荷0-3',
                      'srcurl': [
                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0021/OR021@6_0.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Orthogonal/0021/246.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Opposite/0021/OR021@12_0.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0021/OR021@6_1.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Orthogonal/0021/247.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Opposite/0021/OR021@12_1.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0021/OR021@6_2.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Orthogonal/0021/248.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Opposite/0021/OR021@12_2.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Centered/0021/OR021@6_3.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Orthogonal/0021/249.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Drive End Bearing Fault Data/Outer Race/Opposite/0021/OR021@12_3.mat']]
                      }],

           '12kFE': [{'class': 0, 'classname': 'normal', 'faultname': '基座,健康,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_3.mat']
                      },
                     {'class': 1, 'classname': 'IR007', 'faultname': '风扇端,内圈,故障等级1,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0007/IR007_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0007/IR007_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0007/IR007_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0007/IR007_3.mat']
                      },

                     {'class': 2, 'classname': 'B007', 'faultname': '风扇端,滚珠圈,故障等级1,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0007/B007_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0007/B007_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0007/B007_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0007/B007_3.mat']
                      },

                     {'class': 3, 'classname': 'OR007', 'faultname': '风扇端,外圈,故障等级1,位置@6-3-12,载荷0-3',
                      'srcurl': [[
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0007/298.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_0.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_1.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0007/299.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_1.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_2.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0007/300.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_2.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_3.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0007/301.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_3.mat']]

                      },

                     {'class': 4, 'classname': 'IR014', 'faultname': '风扇端,内圈,故障等级2,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0014/IR014_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0014/IR014_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0014/IR014_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0014/IR014_3.mat']
                      },

                     {'class': 5, 'classname': 'B014', 'faultname': '风扇端,滚珠圈,故障等级2,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0014/B014_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0014/B014_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0014/B014_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0014/B014_3.mat']
                      },

                     {'class': 6, 'classname': 'OR014', 'faultname': '风扇端,外圈,故障等级2,位置@6-3,载荷0-3',
                      'srcurl': [[
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Centered/0014/OR014@6_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0014/310.mat'],
                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0014/311.mat'],
                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0007/312.mat'],
                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0014/313.mat']]
                      },

                     {'class': 7, 'classname': 'IR021', 'faultname': '风扇端,内圈,故障等级3,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0021/IR021_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0021/IR021_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0021/IR021_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Inner Race/0021/IR021_3.mat']
                      },

                     {'class': 8, 'classname': 'B021', 'faultname': '风扇端,滚珠圈,故障等级3,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0021/B021_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0021/B021_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0021/B021_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Ball/0021/B021_3.mat']
                      },

                     {'class': 9, 'classname': 'OR021', 'faultname': '风扇端,外圈,故障等级3,位置@6-3,载荷0-2',
                      'srcurl': [[
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Centered/0021/OR021@6_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0021/316.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0021/317.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/12k Fan End Bearing Fault Data/Outer Race/Orthogonal/0021/318.mat']]
                      }, ],

           '48kDE': [{'class': 0, 'classname': 'normal', 'faultname': '基座,健康,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/Normal Baseline/normal_3.mat']
                      },
                     {'class': 1, 'classname': 'IR007', 'faultname': '48k驱动端,内圈,故障等级1,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0007/IR007_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0007/IR007_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0007/IR007_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0007/IR007_3.mat']
                      },

                     {'class': 2, 'classname': 'B007', 'faultname': '48k驱动端,滚珠圈,故障等级1,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0007/B007_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0007/B007_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0007/B007_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0007/B007_3.mat']
                      },

                     {'class': 3, 'classname': 'OR007', 'faultname': '48k驱动端,外圈,故障等级1,位置@6-3-12,载荷0-3',
                      'srcurl': [[
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Orthogonal/0007/OR007@3_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_0.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_1.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Orthogonal/0007/OR007@3_1.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_1.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_2.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Orthogonal/0007/OR007@3_2.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_2.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0007/OR007@6_3.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Orthogonal/0007/OR007@3_3.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Opposite/0007/OR007@12_3.mat']]
                      },

                     {'class': 4, 'classname': 'IR014', 'faultname': '48k驱动端,内圈,故障等级2,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0014/IR014_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0014/IR014_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0014/IR014_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0014/IR014_3.mat']
                      },

                     {'class': 5, 'classname': 'B014', 'faultname': '48k驱动端,滚珠圈,故障等级2,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0014/B014_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0014/B014_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0014/B014_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0014/B014_3.mat']
                      },

                     {'class': 6, 'classname': 'OR014', 'faultname': '48k驱动端,外圈,故障等级2,位置@6,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0014/OR014@6_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0014/OR014@6_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0014/OR014@6_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0014/OR014@6_3.mat']
                      },

                     {'class': 7, 'classname': 'IR021', 'faultname': '48k驱动端,内圈,故障等级3,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0021/IR021_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0021/IR021_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0021/IR021_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Inner Race/0021/IR021_3.mat']
                      },

                     {'class': 8, 'classname': 'B021', 'faultname': '48k驱动端,滚珠圈,故障等级3,载荷0-3',
                      'srcurl': [
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0021/B021_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0021/B021_1.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0021/B021_2.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Ball/0021/B021_3.mat']
                      },

                     {'class': 9, 'classname': 'OR021', 'faultname': '48k驱动端,外圈,故障等级3,位置@6-3-12,载荷0-3',
                      'srcurl': [[
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0021/OR021@6_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Orthogonal/0021/OR021@3_0.mat',
                          '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Opposite/0021/OR021@12_0.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0021/OR021@6_1.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Orthogonal/0021/OR021@3_1.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Opposite/0021/OR021@12_1.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0021/OR021@6_2.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Orthogonal/0021/OR021@3_2.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Opposite/0021/OR021@12_2.mat'],

                          [
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Centered/0021/OR021@6_3.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Orthogonal/0021/OR021@3_3.mat',
                              '/mnt/E/Projects/Python/Bearing_Fault_Diagnosis/data/raw_data/crwu/48k Drive End Bearing Fault Data/Outer Race/Opposite/0021/OR021@12_3.mat']]
                      }, ]
           }
