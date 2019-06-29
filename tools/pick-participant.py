import random
import sys
from gi.repository import Gtk

class ChangeLabel(Gtk.Window):

    def __init__(self, authors):
        Gtk.Window.__init__(self, title="Presenter")
        self.set_default_size(600, 200)
        self.authors = authors

        table = Gtk.Table(2, 3, True)
        self.add(table)

        self.label = Gtk.Label("Let's start!")
        button = Gtk.Button("Next")
        button.connect("clicked", self.button_pressed)

        table.attach(self.label, 1, 2, 0, 1)
        table.attach(button, 1, 2, 1, 2)

    def button_pressed(self, button):
        try:
            author = self.authors.pop(0)
        except IndexError:
            author = 'No one left'
        # self.label.set_text(author)
        self.label.set_markup("<span font_weight='bold' font='30'>{}</span>".format(author))

author_file = sys.argv[1]
with open(author_file, 'r') as f:
    authors = f.readlines()

random.shuffle(authors)
win = ChangeLabel(authors)
win.connect("delete-event", Gtk.main_quit)
win.show_all()
Gtk.main()
