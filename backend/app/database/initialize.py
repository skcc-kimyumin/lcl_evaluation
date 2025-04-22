from service.model.user import UserRoles
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# def init_role(db: Session):
#     if not db.query(Role).first():
#         initial_roles = [
#             Role(key=UserRoles.common, description="Common user"),
#             Role(key=UserRoles.admin, description="Admin"),
#             Role(key=UserRoles.superadmin, description="Super admin"),
#         ]
#         db.add_all(initial_roles)
#         try:
#             db.commit()
#         except IntegrityError:
#             db.rollback()


# # 초기 데이터 삽입 함수
# def init_db(db: Session):
#     init_role(db)
