import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

class Snapshot():
    graph_list = []

    def __init__(self, fig_shape=(), unify_ylim=False, layout_settings={}):
        self.media_list = []
        self.title_list = []
        if not fig_shape == ():
            self.fig_row = fig_shape[0]
            self.fig_col = fig_shape[1]
        self.use_grid = not layout_settings == {}
        self.use_unified_ylim = unify_ylim
        self.layout_settings = layout_settings
            # expected shape
            # {
            #   'subplot_count': int,
            #   'grid_master': GridSpec Object,
            #   'subplots': [
            #       'subplot_id': int,
            #       'subplot': GridSpecFromSubplotSpec Object
            #   ]
            # }

    def add_media(self, media_type: str, media_data, media_position: int, media_options={}):
        assert media_type in {'image', 'num'}

        self.media_list.append({
            'media_type': media_type,
            'media_data': media_data,
            'media_position': media_position,
            'media_options': media_options
        })

    def get_subplot(self, position: int):
        _media = [x for x in self.media_list     if x['media_position']   == position]
        _graph = [x for x in Snapshot.graph_list if x['position']         == position]
        _title = [x for x in self.title_list     if x['target_media_pos'] == position]

        if len(_media):
            assert len(_media) == 1
            subplot_data = {
                'type': 'media',
                'body': _media[0],
                'title': {}
            }
            if len(_title):
                assert len(_title) == 1
                subplot_data['title'] = _title[0]
            return subplot_data

        elif len(_graph):
            assert len(_graph) == 1
            subplot_data = {
                'type': 'graph',
                'body': _graph[0],
                'title': {}
            }
            if len(_title):
                assert len(_title) == 1
                subplot_data['title'] = _title[0]
            return subplot_data

        else:
            raise TypeError('No subplot found in position')

    @classmethod
    def add_graph_data(self, graph_id: str, data_id: str, new_data, frame_num: int):
        target_graph_index = -1
        target_graph_data = {}
        target_data_index = -1

        for i, graph in enumerate(Snapshot.graph_list):
            if graph['id'] == graph_id:
                target_graph_index = i
                target_graph_data = graph
                break

        for i, data in enumerate(target_graph_data['data']):
            if data['data_id'] == data_id:
                target_data_index = i

        if target_graph_index >= 0:
            if frame_num >= Snapshot.graph_list[target_graph_index]['settings']['frame_in_rotation']:
                raise TypeError('Frame number not in formerly specified frame maximum.')

            if target_data_index >= 0:
                Snapshot.graph_list[target_graph_index]['data'][target_data_index]['frame_data'][frame_num] = new_data

            else:
                Snapshot.graph_list[target_graph_index]['data'].append({
                    'data_id': data_id,
                    'frame_data': [
                        [] for j in range(Snapshot.graph_list[target_graph_index]['settings']['frame_in_rotation'])
                    ]
                })
                Snapshot.graph_list[target_graph_index]['data'][-1]['frame_data'][frame_num] = new_data

        else:
            raise TypeError('Graph with specified id does not exist. Check your id or if you have called make_plt()')

    def add_title(self, text: str, target_media_pos: int, title_options={}):
        self.title_list.append({
            'text': text,
            'target_media_pos': target_media_pos,
            'title_options': title_options
        })

    @classmethod
    def make_graph(self, id: str,
                       pos: int,
                       graph_type: str,
                       frame_in_rotation: int,
                       num_of_data_per_graph=1,
                       layout_settings={},
                       trivial_settings={}):

        for graph in Snapshot.graph_list:
            if graph['id'] == id:
                return

        Snapshot.graph_list.append({
            'id': id,
            'data': [
                # expected shape
                # {
                #     'data_id': '',
                #     'frame_data': [
                #         [] for j in range(frame_in_rotation)
                #     ]
                # }
            ],
            'position': pos,
            'settings': {
                'type': graph_type,
                'frame_in_rotation': frame_in_rotation,
                'num_of_data_per_graph': num_of_data_per_graph
            },
            'sub_settings': trivial_settings
        })

    def print_to_fig(self, fig, frame_num):
        if self.use_grid:
            position_range = self.layout_settings['subplot_count']
        else:
            position_range = int(self.fig_row * self.fig_col)

        if self.use_unified_ylim:
            y_max = max([max(d['frame_data']) for graph in Snapshot.graph_list for d in graph['data']])

        for i in range(position_range):
            position = i + 1
            _media = [x for x in self.media_list     if x['media_position']   == position]
            _graph = [x for x in Snapshot.graph_list if x['position']         == position]
            _title = [x for x in self.title_list     if x['target_media_pos'] == position]

            if self.use_grid:
                target_subplot_axis_list = [subplot['subplot']
                            for subplot in self.layout_settings['subplots']
                            if subplot['subplot_id'] == position ]

                if len(target_subplot_axis_list):
                    target_subplot = target_subplot_axis_list[0]
                else:
                    continue

                axis = fig.add_subplot(target_subplot[:, :])
            else:
                if len(fig.axes) >= position and len(_graph):
                    fig.delaxes(fig.axes[position-1])
                axis = fig.add_subplot(self.fig_row, self.fig_col, position)

            if len(_media):
                if len(_media) > 1:
                    raise TypeError('Multiple media located on same position')

                media = _media[0]
                if media['media_type'] == 'image':
                    axis.axis('off')
                    axis.imshow(media['media_data'], interpolation="none", animated=True)

                else:
                    assert isinstance(media['media_options']['coordinates'], tuple)
                    pos_x, pos_y = media['media_options']['coordinates']

                    axis.axis('off')
                    axis.text(pos_x, pos_y, 'KL_Div = {:.3f}'.format(media['media_data']))

                if len(_title):
                    if len(_title) > 1:
                        raise TypeError('Multiple title located on same position')
                    title = _title[0]
                    axis.set_title(title['text'])

            elif len(_graph):
                if len(_graph) > 1:
                    raise TypeError('Multiple graph located on same posiiton')

                graph = _graph[0]

                graph_type = graph['settings']['type']
                plt_settings = graph['sub_settings']

                if graph_type == 'plot':
                    axis.set_xlim(1, graph['settings']['frame_in_rotation'])

                    if not self.use_unified_ylim:
                        y_max = max([max([y for y in x['frame_data']]) for x in graph['data']]) * 1.1
                    axis.set_ylim(0, y_max)

                    if 'noXTicks' in plt_settings:
                        axis.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    if 'noYTicks' in plt_settings:
                        axis.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                    if 'xscale' in plt_settings:
                        if plt_settings['xscale'] in ['linear', 'log', 'symlog', 'logit']:
                            axis.set_xscale(plt_settings['xscale'])
                    if 'yscale' in plt_settings:
                        if plt_settings['yscale'] in ['linear', 'log', 'symlog', 'logit']:
                            axis.set_yscale(plt_settings['yscale'])
                    if 'colors' not in plt_settings:
                        raise TypeError('Plotting requires "colors" setting in trivial settings')
                    if not len(plt_settings['colors']) == graph['settings']['num_of_data_per_graph']:
                        raise TypeError('Number of colors specified does not match the number of data to be drawn')
                    if 'markers' not in plt_settings:
                        raise TypeError('Plotting requires "markers" setting in trivial settings')
                    if not len(plt_settings['markers']) == graph['settings']['num_of_data_per_graph']:
                        raise TypeError('Number of markers specified does not match the number of data to be drawn')
                    _color = plt_settings['colors']
                    _marker = plt_settings['markers']

                    last_data_num = int(frame_num / graph['settings']['frame_in_rotation'])
                    last_frame_num = frame_num % graph['settings']['frame_in_rotation'] + 1

                    for data_num in range(last_data_num + 1):
                        frame_array = []
                        if data_num == last_data_num:
                            frame_array = np.arange(1, last_frame_num + 1, 1)
                            axis.plot(frame_array,
                                    graph['data'][data_num]['frame_data'][:last_frame_num],
                                    color=_color[data_num],
                                    marker=_marker[data_num])

                        else:
                            frame_array = np.arange(1, graph['settings']['frame_in_rotation'] + 1, 1)
                            axis.plot(frame_array,
                                    graph['data'][data_num]['frame_data'],
                                    color=_color[data_num],
                                    marker=_marker[data_num])

                if len(_title):
                    if len(_title) > 1:
                        raise TypeError('Multiple title located on same position')
                    title = _title[0]
                    axis.set_title(title['text'])

                else:
                    pass #still not made

            else:
                axis.axis('off')