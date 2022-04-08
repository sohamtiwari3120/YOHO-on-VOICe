from audioop import add
from config import add_EAP_to_path
add_EAP_to_path()
import model.attention.CoAtNet as CN
def main():
    model = CN()
    print(model)

if __name__ == '__main__':
    main()