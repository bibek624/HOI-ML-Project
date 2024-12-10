
import json


class Image:
    file_name: str
    url: str
    height: int
    width: int
    id: int

    def __init__(self, file_name: str, url: str, height: int, width: int, id: int) -> None:
        self.file_name = file_name
        self.url = url
        self.height = height
        self.width = width
        self.id = id
        
        self.categories = []
        self.catIds = []
        self.bboxes = []
        self.isCrowds = []
        

    @staticmethod
    def from_dict(obj) -> 'Image':
        assert isinstance(obj, dict)
        file_name = obj.get("file_name")
        url = obj.get("coco_url")
        height = obj.get("height")
        width = obj.get("width")
        id = obj.get("id")
        return Image(file_name, url, height, width, id)


        
class Data():
    
    def __init__(self, filepath):
        self.images = None
        with open(filepath,'rt') as file:
            self.data = json.load(file)
            
        self.load_data()
        

    def load_data(self):
        
        self.images = [Image.from_dict(image) for image in self.data['images']]    
        
        for image in self.images:
            for annotation in self.data['annotations']:
                if annotation['image_id'] == image.id:
                    image.catIds.append(annotation['category_id'])
                    image.bboxes.append(annotation['bbox'])
                    image.isCrowds.append(annotation['iscrowd'])       
        
# data = Data('instances_vcoco_all_2014.json')

# print(data.images)

filepath = 'instances_vcoco_all_2014.json'
with open(filepath,'rt') as file:
    coco_data = json.load(file)
        

with open('vcoco_train.json','rt') as file:
    vcoco_data = json.load(file)
    

images = []

for image in coco_data['images'][:100]:
    
    data_dict = {'id': 0, 'file_name':'', 'url':'','height':0,'width':0, 'categories':[], 'bboxes':[], 'isCrowds':[]}
    
    data_dict['id'] = image.get("id")
    data_dict["file_name"] = image.get("file_name")
    data_dict['url'] = image.get("coco_url")
    data_dict['height'] = image.get("height")
    data_dict['width'] = image.get("height")
    
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image['id']:
            data_dict['categories'].append(annotation['category_id'])
            data_dict['bboxes'].append(annotation['bbox'])
            data_dict['isCrowds'].append(annotation['iscrowd'])
    
    images.append(data_dict)
print(images[0])
#     self.images = [Image.from_dict(image) for image in self.data['images']]    
    
#     for image in self.images:
#         for annotation in self.data['annotations']:
#             if annotation['image_id'] == image.id:
#                 image.catIds.append(annotation['category_id'])
#                 image.bboxes.append(annotation['bbox'])
#                 image.isCrowds.append(annotation['iscrowd'])       
        
# data = Data('instances_vcoco_all_2014.json')

        