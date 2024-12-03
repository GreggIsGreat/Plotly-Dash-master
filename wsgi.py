from final_dashboard import app

if __name__ == '__main__':
    from gunicorn.app.wsgiapp import run
    run(app.server)