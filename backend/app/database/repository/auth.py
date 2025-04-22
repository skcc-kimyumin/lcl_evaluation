from datetime import datetime

from database.repository.user import get_user_by_user_id
from sqlalchemy.orm import Session


def update_last_sign_in(db: Session, user_id: str):
    user = get_user_by_user_id(db, user_id)
    user.last_sign_in = datetime.utcnow()
    db.commit()
