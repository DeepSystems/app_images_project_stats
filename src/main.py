import supervisely_lib as sly

my_app = sly.AppService()


@my_app.callback("generate")
@sly.timeit
def generate_random_string(api: sly.Api, task_id, context, state):
    new_str = sly.rand_str(10)
    api.app.set_vars(task_id, "data.randomString", new_str)


def main():
    # data
    data = {
        "randomString": "initial random value xxx"
    }

    # state
    state = {
    }

    # Example:
    my_app.run(data=data, state=state)


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