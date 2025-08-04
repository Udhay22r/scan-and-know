from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User, ScanHistory
import json

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
    
    return render_template('login.html')

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'})
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'})
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': 'Registration failed'})
    
    return render_template('register.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@auth.route('/dashboard')
@login_required
def dashboard():
    # Get user's scan history
    scan_history = ScanHistory.query.filter_by(user_id=current_user.id).order_by(ScanHistory.scan_date.desc()).limit(10).all()
    return render_template('dashboard.html', scan_history=scan_history)

@auth.route('/api/save_scan', methods=['POST'])
@login_required
def save_scan():
    data = request.get_json()
    
    scan_history = ScanHistory(
        user_id=current_user.id,
        product_name=data.get('product_name', 'Unknown Product'),
        barcode=data.get('barcode'),
        product_data=json.dumps(data.get('product_data', {})),
        scan_type=data.get('scan_type', 'unknown')
    )
    
    try:
        db.session.add(scan_history)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': 'Failed to save scan'})

@auth.route('/api/scan_history')
@login_required
def get_scan_history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    scan_history = ScanHistory.query.filter_by(user_id=current_user.id)\
        .order_by(ScanHistory.scan_date.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    history_data = []
    for scan in scan_history.items:
        history_data.append({
            'id': scan.id,
            'product_name': scan.product_name,
            'barcode': scan.barcode,
            'scan_date': scan.scan_date.strftime('%Y-%m-%d %H:%M'),
            'scan_type': scan.scan_type,
            'product_data': json.loads(scan.product_data) if scan.product_data else {}
        })
    
    return jsonify({
        'history': history_data,
        'has_next': scan_history.has_next,
        'has_prev': scan_history.has_prev,
        'page': page
    }) 