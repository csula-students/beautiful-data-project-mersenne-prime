from source import source
from collector import collector

so = source("datasets/TUT-acoustic-scenes-2016-development/")

so.source_save()

print(co.get_scenes())

print(co.get_feature_vector_array(sound_scene="car",limit_num=1))
