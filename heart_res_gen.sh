python main.py --dataset   heart --activation sigmoid --optimizer momentum --gradient batch --learning-rate 0.01 --epochs 100
python main.py --dataset   heart --activation sigmoid --optimizer none --gradient batch --learning-rate 0.05 --epochs 100
python main.py --dataset   heart --activation sigmoid --optimizer momentum --gradient minibatch --learning-rate 0.01 --epochs 500
python main.py --dataset   heart --activation sigmoid --optimizer none --gradient minibatch --learning-rate 0.05 --epochs 500
python main.py --dataset   heart --activation sigmoid --optimizer momentum --gradient stochastic --learning-rate 0.05 --epochs 1500
python main.py --dataset   heart --activation sigmoid --optimizer none --gradient stochastic --learning-rate 0.08 --epochs 2500

python main.py --dataset   heart --activation tanh --optimizer momentum --gradient batch --learning-rate 0.01 --epochs 100
python main.py --dataset   heart --activation tanh --optimizer none --gradient batch --learning-rate 0.05 --epochs 100
python main.py --dataset   heart --activation tanh --optimizer momentum --gradient minibatch --learning-rate 0.008 --epochs 500
python main.py --dataset   heart --activation tanh --optimizer none --gradient minibatch --learning-rate 0.01 --epochs 500
python main.py --dataset   heart --activation tanh --optimizer momentum --gradient stochastic --learning-rate 0.01 --epochs 1000
python main.py --dataset   heart --activation tanh --optimizer none --gradient stochastic --learning-rate 0.05 --epochs 1500

python main.py --dataset   heart --activation relu --optimizer momentum --gradient batch --learning-rate 0.008 --epochs 100
python main.py --dataset   heart --activation relu --optimizer none --gradient batch --learning-rate 0.05 --epochs 100
python main.py --dataset   heart --activation relu --optimizer momentum --gradient minibatch --learning-rate 0.008 --epochs 500
python main.py --dataset   heart --activation relu --optimizer none --gradient minibatch --learning-rate 0.07 --epochs 500
python main.py --dataset   heart --activation relu --optimizer momentum --gradient stochastic --learning-rate 0.005 --epochs 1000
python main.py --dataset   heart --activation relu --optimizer none --gradient stochastic --learning-rate 0.005 --epochs 1800