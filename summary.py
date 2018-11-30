from tensorboardX import SummaryWriter

# import visualize
writer = SummaryWriter()


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_summary(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        self.writer.add_image(tag, images, step)
    
    def graph_summary(self, model, dummy_input):
        self.writer.add_graph(model,(dummy_input,))
        
