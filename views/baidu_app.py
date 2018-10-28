import base64
import json
import random

from flask import Flask, request, render_template

def registe_route(app):
    @app.route('/baidu/<id>')
    def baidu_index(id):
        return render_template("baidu.html", id=id, randomKey=random.random())


