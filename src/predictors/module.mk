MODULE := src/predictors

MODULE_OBJS := \
	src/predictors/cnnAE_model_loader.o \
	src/predictors/cnnAE.o \

MODULE_DIRS += \
	src/predictors

# Include common rules 
include $(srcdir)/common.rules
