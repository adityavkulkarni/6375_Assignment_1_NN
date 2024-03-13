python main.py --dataset  cancer --activation sigmoid --optimizer momentum --gradient batch --learning-rate 0.005 --epochs 50
python main.py --dataset  cancer --activation sigmoid --optimizer none --gradient batch --learning-rate 0.001 --epochs 50
python main.py --dataset cancer --activation sigmoid --optimizer momentum --gradient minibatch --learning-rate 0.001 --epochs 100
python main.py --dataset  cancer --activation sigmoid --optimizer none --gradient minibatch --learning-rate 0.005 --epochs 100
python main.py --dataset  cancer --activation sigmoid --optimizer momentum --gradient stochastic --learning-rate 0.005 --epochs 500
python main.py --dataset  cancer --activation sigmoid --optimizer none --gradient stochastic --learning-rate 0.005 --epochs 1000

python main.py --dataset  cancer --activation tanh --optimizer momentum --gradient batch --learning-rate 0.001 --epochs 50
python main.py --dataset  cancer --activation tanh --optimizer none --gradient batch --learning-rate 0.005 --epochs 50
python main.py --dataset  cancer --activation tanh --optimizer momentum --gradient minibatch --learning-rate 0.005 --epochs 500
python main.py --dataset  cancer --activation tanh --optimizer none --gradient minibatch --learning-rate 0.001 --epochs 500
python main.py --dataset  cancer --activation tanh --optimizer momentum --gradient stochastic --learning-rate 0.001 --epochs 500
python main.py --dataset  cancer --activation tanh --optimizer none --gradient stochastic --learning-rate 0.005 --epochs 800

python main.py --dataset  cancer --activation relu --optimizer momentum --gradient batch --learning-rate 0.005 --epochs 50
python main.py --dataset  cancer --activation relu --optimizer none --gradient batch --learning-rate 0.005 --epochs 50
python main.py --dataset  cancer --activation relu --optimizer momentum --gradient minibatch --learning-rate 0.005 --epochs 500
python main.py --dataset  cancer --activation relu --optimizer none --gradient minibatch --learning-rate 0.005 --epochs 100
python main.py --dataset  cancer --activation relu --optimizer momentum --gradient stochastic --learning-rate 0.005 --epochs 100
python main.py --dataset  cancer --activation relu --optimizer none --gradient stochastic --learning-rate 0.005 --epochs 1800