from collections import defaultdict
import numpy as np

import supervisely_lib as sly

my_app = sly.AppService()

BG_COLOR = [0, 0, 0]

@my_app.callback("calculate")
@sly.timeit
def calculate(api: sly.Api, task_id, context, state):
    project_id = 11#context.get("project_id")
    project = api.project.get_info_by_id(project_id)
    if project is None:
        raise RuntimeError("Project ID={!r} not found".format(project_id))

    #@TODO: uncomment
    #if project_id is None:
    #    sly.logger.critical("Project ID nor found")

    meta_json = api.project.get_meta(project_id)
    meta = sly.ProjectMeta.from_json(meta_json)

    # list classes
    class_names = []
    class_colors = []
    class_indices = []  # 0 - for unlabeled area
    class_indices_colors = []
    _name_to_index = {}
    for idx, obj_class in enumerate(meta.obj_classes):
        class_names.append(obj_class.name)
        class_colors.append(obj_class.color)
        class_index = idx + 1
        class_indices.append(class_index)
        class_indices_colors.append([class_index, class_index, class_index])
        _name_to_index[obj_class.name] = class_index

    # list tags
    tag_names = []
    tag_colors = []
    for tag_meta in meta.tag_metas:
        tag_names.append(tag_meta.name)
        tag_colors.append(tag_meta.color)

    # check colors uniq
    color_names = defaultdict(list)
    for name, color in zip(class_names, class_colors):
        hex = sly.color.rgb2hex(color)
        color_names[hex].append(name)

    class_colors_notify = ""
    for k, v in color_names.items():
        if len(v) > 1:
            warn_str = "Classes {!r} have the same RGB color = {!r}".format(v, sly.color.hex2rgb(k))
            sly.logger.warn(warn_str)
            class_colors_notify += warn_str + '\n\n'

    for dataset in api.dataset.get_list(project.id):
        images = api.image.get_list(dataset.id)
        for batch in sly.batched(images):
            image_ids = [image_info.id for image_info in batch]
            image_names = [image_info.name for image_info in batch]

            ann_infos = api.annotation.download_batch(dataset.id, image_ids)
            ann_jsons = [ann_info.annotation for ann_info in ann_infos]

            for info, ann_json in zip(batch, ann_jsons):
                ann = sly.Annotation.from_json(ann_json, meta)

                render_img = np.zeros(ann.img_size + (3,), dtype=np.uint8)
                render_img[:] = BG_COLOR
                ann.draw(render_img)
                # temp_area = sly.Annotation.stat_area(render_img, class_names, class_colors)

                render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
                render_idx_rgb[:] = BG_COLOR
                ann.draw_class_idx_rgb(render_idx_rgb, _name_to_index)
                temp_area = sly.Annotation.stat_area(render_idx_rgb, class_names, class_indices_colors)

                temp_count = ann.stat_class_count(class_names)
                if len(tag_names) != 0:
                    temp_img_tags = ann.stat_img_tags(tag_names)

                temp_area['id'] = info.id
                temp_area['dataset'] = dataset.name
                temp_area['name'] = '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>' \
                    .format(api.image.url(team.id,
                                          workspace.id,
                                          project.id,
                                          dataset.id,
                                          info.id),
                            info.name)

                stats_area.append(temp_area)
                stats_count.append(temp_count)
                if len(tag_names) != 0:
                    stats_img_tags.append(temp_img_tags)

            ds_progress.iters_done_report(len(batch))
            total_images_in_project += len(batch)
    #@TODO: implement here
    #if class_colors_notify != "":
    #    widgets.append(
    #        api.report.create_notification("Classes colors", class_colors_notify, sly.NotificationType.WARNING))

    print("Hello!")
    # new_str = sly.rand_str(10)
    # api.app.set_vars(task_id, "data.randomString", new_str)


#@TODO: show warning
def main():
    table = []
    for i in range(10):
        table.append({"name": sly.rand_str(5), "my_value": i})

    # data
    data = {
        "table": table
    }

    # state
    state = {
        "perPage": 25,
        "pageSizes": [25, 50, 100]
    }

    # start event after successful service run
    events = [
        {
            "state": {},
            "context": {},
            "command": "calculate"
        }
    ]

    # Run application service
    my_app.run(data=data, state=state, initial_events=events)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sly.logger.critical('Unexpected exception in main.', exc_info=True, extra={
            'event_type': sly.EventType.TASK_CRASHED,
            'exc_str': str(e),
        })

#@TODO:
# python -m pip install git+https://github.com/supervisely/supervisely
# python setup.py develop
# context + state по всем юзерам? + там будет labelerLogin, api_token, и тд