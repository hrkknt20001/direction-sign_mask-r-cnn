import urllib.request
import json
import os
import io
import numpy as np
from PIL import Image, ImageDraw

 
def create_opener():
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('Authorization', 'Token xxxxxxxxxxxxxxxxxxxxxxxxxxx')
    ]
    return opener

def download_image(url, dst_path=None):
    opener = create_opener()
    with opener.open(url) as web_file:
        data = web_file.read()
        img = Image.open(io.BytesIO(data))
        if dst_path is not None:
            img.save(dst_path)

    return img

try:
    opener = create_opener()

    for task_no in range(825, 904):
        with opener.open(f'http://localhost:8080/api/tasks/{task_no}/') as response:
            body = json.loads(response.read())
            headers = response.getheaders()
            status = response.getcode()

            #print(headers)
            #print(status)
            #print(body)

        if len(body['annotations']) == 0:
            continue

        url = body['data']['image']
        image = download_image( 'http://localhost:8080' + url )

        print(os.path.basename(url))

        img = Image.new("RGB", (image.width, image.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        count = 0
        for idx, result in enumerate(body['annotations'][0]['result']):
            if result['type'] == 'rectanglelabels':
                org_width = result['original_width']
                org_height = result['original_height']

                if result['value']['rectanglelabels'][0] == 'text':
                    x = result['value']['x']
                    y = result['value']['y']
                    width = result['value']['width']
                    height = result['value']['height']
                    rotation = result['value']['rotation']

                    points = (
                        ( (x        ) / 100 * org_width, (y         ) / 100 * org_height ),
                        ( (x + width) / 100 * org_width, (y         ) / 100 * org_height ),
                        ( (x + width) / 100 * org_width, (y + height) / 100 * org_height ),
                        ( (x        ) / 100 * org_width, (y + height) / 100 * org_height ),
                        ( (x        ) / 100 * org_width, (y         ) / 100 * org_height )
                    )

                    draw.polygon( points, fill=(idx+1, idx+1, idx+1), outline=(idx+1, idx+1, idx+1))
                    count += 1

        if count != 0:
            p = os.path.splitext(os.path.basename(url))[0].find('-')
            basename = os.path.splitext(os.path.basename(url))[0][p+1:]
            
            img = img.convert("L")
            img.save( 
                os.path.join(
                    'data', 'Masks', f'{basename}_mask.png'
                )
            )

            image.save( 
                os.path.join(
                    'data', 'Images', f'{basename}.png'
                )
            )


    print('done')

except urllib.error.URLError as e:
    print(e.reason)

