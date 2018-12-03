class Snapshot():
    def __init__(self, fig_shape):
        self.media_list = []
        self.title_list = []
        self.fig_row = fig_shape[0]
        self.fig_col = fig_shape[1]

    def add_media(self, media_type: str, media_data, media_position: int, media_options=[]):
        assert media_type in {'image', 'plt'}
        assert media_position <= self.fig_row * self.fig_col

        self.media_list.append({
            'media_type': media_type,
            'media_data': media_data,
            'media_position': media_position,
            'media_options': media_options
        })

    def add_title(self, text: str, target_media_pos: int, title_options=[]):
        self.title_list.append({
            'text': text,
            'target_media_pos': target_media_pos,
            'title_options': title_options
        })

    def print_to_fig(self, fig):
        for i in range(int(self.fig_row * self.fig_col)):
            position = i + 1
            _media = [x for x in self.media_list if x['media_position'] == position]
            _title = [x for x in self.title_list if x['target_media_pos'] == position]

            if len(_media):
                media = _media[0]
                axis = fig.add_subplot(self.fig_row, self.fig_col, media['media_position'])

                if media['media_type'] == 'image':
                    axis.axis('off')
                    axis.imshow(media['media_data'], interpolation="none", animated=True)
                else:
                    pass

                if len(_title):
                    title = _title[0]
                    axis.set_title(title)