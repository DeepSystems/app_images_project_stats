from collections import defaultdict
import numpy as np

import supervisely_lib as sly

my_app = sly.AppService()

BG_COLOR = [0, 0, 0]


def area_name(name):
    return "{} [area %]".format(name)


def count_name(name):
    return "{} [count]".format(name)


def color_name(name, color):
    return '<b style="display: inline-block; border-radius: 50%; background: {}; width: 8px; height: 8px"></b> {}'.format(color, name)


def color_tag_name(name, color):
    return '<i class="zmdi zmdi-label" style="color:{};margin-right:3px"></i>{}'.format(sly.color.rgb2hex(color), name)


def rename_fields(data, old_names, new_name_func, colors=None, color_func=None):
    colors = sly.take_with_default(colors, [None] * len(old_names))
    for name, color in zip(old_names, colors):
        if name in data:
            new_name = new_name_func(name)
            if color is not None and color_func is not None:
                new_name = color_func(new_name, color)
            data[new_name] = data[name]
            del data[name]


@my_app.callback("calculate")
@sly.timeit
def calculate(api: sly.Api, task_id, context, state):
    project_id = 502 #context.get("project_id")
    project = api.project.get_info_by_id(project_id)
    if project is None:
        raise RuntimeError("Project ID={!r} not found".format(project_id))
    if project.type != str(sly.ProjectType.IMAGES):
        raise RuntimeError('Project {!r} has type {!r}. This script works only with {!r} projects'
                           .format(project.name, project.type, str(sly.ProjectType.IMAGES)))

    workspace = api.workspace.get_info_by_id(project.workspace_id)
    team = api.team.get_info_by_id(workspace.team_id)

    sly.logger.info("team: {}".format(team.name))
    sly.logger.info("workspace: {}".format(workspace.name))
    sly.logger.info("project: {}".format(project.name))

    meta_json = api.project.get_meta(project_id)
    meta = sly.ProjectMeta.from_json(meta_json)

    # list classes (used when several classes have the same colors )
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

    total_images_count = api.project.get_images_count(project.id)

    #total_images_in_project = 0
    #stats_area = []
    #stats_count = []
    #stats_img_tags = []

    table_per_image_stats = []
    table_batch = []
    for dataset in api.dataset.get_list(project.id):
        images = api.image.get_list(dataset.id)
        ds_progress = sly.Progress('Dataset {}'.format(dataset.name), total_cnt=len(images))

        for batch in sly.batched(images):
            image_ids = [image_info.id for image_info in batch]
            #image_names = [image_info.name for image_info in batch]

            ann_infos = api.annotation.download_batch(dataset.id, image_ids)
            ann_jsons = [ann_info.annotation for ann_info in ann_infos]

            for info, ann_json in zip(batch, ann_jsons):
                ann = sly.Annotation.from_json(ann_json, meta)

                render_img = np.zeros(ann.img_size + (3,), dtype=np.uint8)
                render_img[:] = BG_COLOR
                ann.draw(render_img)

                render_idx_rgb = np.zeros(ann.img_size + (3,), dtype=np.uint8)
                render_idx_rgb[:] = BG_COLOR
                ann.draw_class_idx_rgb(render_idx_rgb, _name_to_index)

                temp_area = sly.Annotation.stat_area(render_idx_rgb, class_names, class_indices_colors, percent=True)
                rename_fields(temp_area, class_names, area_name, class_colors, color_name)

                temp_count = ann.stat_class_count(class_names)
                rename_fields(temp_count, class_names, count_name, class_colors, color_name)

                temp_img_tags = {}
                if len(tag_names) != 0:
                    temp_img_tags = ann.stat_img_tags(tag_names)
                    rename_fields(temp_img_tags, tag_names, lambda x: x, tag_colors, color_tag_name)

                temp_area['id'] = info.id
                temp_area['dataset'] = dataset.name
                temp_area['name'] = '<a href="{0}" rel="noopener noreferrer" target="_blank">{1}</a>' \
                    .format(api.image.url(team.id, workspace.id, project.id, dataset.id, info.id), info.name)

                table_batch.append({**temp_area, **temp_count, **temp_img_tags})

            ds_progress.iters_done_report(len(batch))
            table_per_image_stats.extend(table_batch)

            # refresh table and progress
            payload = {
                "progress": int(len(table_per_image_stats) / total_images_count * 100),
                "tablePerImageStats": table_batch
            }
            api.app.set_data(task_id, payload, "data", append=True)


def main():
    table = []
    for i in range(10):
        table.append({"name": sly.rand_str(5), "my_value": i})

    # data
    data = {
        "tablePerImageStats": table,
        "progress": 0
    }

    # state
    state = {
        "perPage": 25,
        "pageSizes": [25, 50, 100],
        "processingFlag": True
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